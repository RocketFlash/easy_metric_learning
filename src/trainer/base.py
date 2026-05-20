from tqdm.auto import tqdm
import torch
from easydict import EasyDict as edict

from ..utils import AverageMeter
from ..model.margin.utils import get_incremental_margin
from ..transform import mix_transform
from ..loss.mix import MixCriterion
from ..loss import get_loss
from ..visualization import save_batch_grid
from ..scheduler import (
    get_scheduler,
    get_warmup_scheduler,
    scheduler_steps_per_batch,
    set_scheduler_tmax,
    step_scheduler,
)
from .margin_regularization import add_margin_regularization_loss
from .loss_inputs import calculate_weighted_loss
from .targets import get_keypoints, move_to_device, split_targets
from .dpap import apply_dpap_transform, get_dpap_transform
from .model_averaging import (
    averaged_model_has_updates,
    create_averaged_model,
    should_update_averaged_model,
    should_use_averaged_model_for_eval,
)
from .xbm import create_xbm, get_xbm_for_targets, update_xbm


def _amp_device_type(device):
    device_type = device.type if isinstance(device, torch.device) else str(device)
    device_type = device_type.split(":")[0]
    if device_type == "cpu":
        return None
    if device_type.isdigit():
        return "cuda"
    return device_type


class BaseTrainer:
    def __init__(
        self,
        config,
        model,
        optimizer,
        epoch=1,
        work_dir="./",
        device="cpu",
        ids_to_labels=None,
    ):

        self.config = config
        self.model = model
        self.epoch = epoch
        self.device = device
        self.work_dir = work_dir
        self.optimizer = optimizer
        self.ids_to_labels = ids_to_labels

        self.n_epochs = config.epochs - epoch + 1
        self.loss_fns = get_loss(loss_config=config.loss, device=device)
        self.amp_device_type = _amp_device_type(device)
        self.amp_scaler = (
            torch.amp.GradScaler(self.amp_device_type)
            if self.amp_device_type is not None and config.amp
            else None
        )

        set_scheduler_tmax(config.scheduler.scheduler, self.n_epochs)

        self.scheduler = get_scheduler(
            self.optimizer, scheduler_config=config.scheduler
        )
        self.warmup_scheduler = get_warmup_scheduler(
            self.optimizer, scheduler_config=config.scheduler
        )
        self.scheduler_step_per_batch = scheduler_steps_per_batch(self.scheduler)

        self.debug = config.debug
        self.visualize_batch = config.visualize_batch
        self.grad_accum_steps = config.train.trainer.grad_accum_steps
        self.averaged_model = create_averaged_model(config, model, device=device)
        self.xbm = create_xbm(config, device=device)

        self.mix_loss_fns = {
            k: edict(
                {
                    "loss_fn": (
                        MixCriterion(v.loss_fn)
                        if getattr(v, "mixable", True)
                        else v.loss_fn
                    ),
                    "weight": v.weight,
                    "input": getattr(v, "input", "output"),
                    "mixable": getattr(v, "mixable", True),
                }
            )
            for k, v in self.loss_fns.items()
        }
        self.dpap_transform = get_dpap_transform(config)

        incremental_margin_config = getattr(config.margin, "incremental_margin", None)
        if incremental_margin_config is not None:
            if not hasattr(self.model.margin, "m") or not hasattr(
                self.model.margin, "update"
            ):
                raise ValueError(
                    f"{config.margin.type} does not support incremental_margin"
                )
            self.incremental_margin = get_incremental_margin(
                m_max=self.model.margin.m,
                m_min=incremental_margin_config.min_m,
                n_epochs=config.epochs,
                mode=incremental_margin_config.type,
            )
        else:
            self.incremental_margin = None

    def _get_margin_value(self):
        margin = getattr(self.model, "margin", None)
        margin_value = getattr(margin, "m", None)
        if isinstance(margin_value, tuple):
            return f"{margin_value[0]}..{margin_value[1]}"
        return margin_value

    def _get_labels_for_visualization(self, targets):
        if isinstance(targets, (list, tuple)):
            return targets[0]
        return targets

    def _forward_model(self, images, targets, target_extras, model=None):
        model = self.model if model is None else model
        keypoints = get_keypoints(target_extras)
        if keypoints is not None:
            return model(images, targets, keypoints=keypoints)
        return model(images, targets)

    def _apply_dpap(self, images):
        return apply_dpap_transform(self.dpap_transform, images)

    def _update_averaged_model(self):
        if self.averaged_model is None:
            return
        if should_update_averaged_model(self.config, self.epoch):
            self.averaged_model.update_parameters(self.model)

    def _step_scheduler(self, metric=None):
        if self.warmup_scheduler is not None:
            with self.warmup_scheduler.dampening():
                step_scheduler(self.scheduler, metric)
        else:
            step_scheduler(self.scheduler, metric)

    def get_eval_model(self):
        if should_use_averaged_model_for_eval(
            self.config
        ) and averaged_model_has_updates(self.averaged_model):
            return self.averaged_model.module
        return self.model

    def train_epoch(self, train_loader):
        if hasattr(self.optimizer, "train"):
            self.optimizer.train()
        self.model.train()

        if self.incremental_margin is not None:
            self.model.margin.update(self.incremental_margin[self.epoch - 1])

        loss_meters = {k: AverageMeter() for k, v in self.loss_fns.items()}
        loss_meters["total_loss"] = AverageMeter()

        tqdm_train = tqdm(train_loader, total=int(len(train_loader)))

        for batch_index, (images, targets, file_names) in enumerate(tqdm_train):
            if self.debug and batch_index >= 10:
                break

            targets, target_extras = split_targets(targets)

            images, targets, is_mixed = mix_transform(
                images,
                targets,
                cutmix_p=self.config.transform.cutmix.p,
                cutmix_alpha=self.config.transform.cutmix.alpha,
                mixup_p=self.config.transform.mixup.p,
                mixup_alpha=self.config.transform.mixup.alpha,
            )

            if is_mixed:
                criterion = self.mix_loss_fns
                target_extras = {}
                targets[0] = targets[0].to(self.device)
                targets[1] = targets[1].to(self.device)
            else:
                criterion = self.loss_fns
                targets = targets.to(self.device)
                target_extras = move_to_device(target_extras, self.device)

            images = images.to(self.device)
            images = self._apply_dpap(images)

            if self.visualize_batch:
                if self.epoch == 1 and batch_index == 0:
                    vis_targets = self._get_labels_for_visualization(targets)
                    labels = [self.ids_to_labels[anno.item()] for anno in vis_targets]
                    save_batch_grid(
                        images.cpu(),
                        labels,
                        self.config.backbone.norm_std,
                        self.config.backbone.norm_mean,
                        save_dir=self.work_dir,
                        split="train",
                    )

            total_loss = 0
            if self.amp_scaler is not None:
                with torch.amp.autocast(self.amp_device_type):
                    output, emb = self._forward_model(images, targets, target_extras)
                    xbm = get_xbm_for_targets(self.xbm, targets)

                    for loss_name, loss_params in criterion.items():
                        loss = calculate_weighted_loss(
                            loss_params,
                            output,
                            emb,
                            targets,
                            images=images,
                            xbm=xbm,
                        )
                        loss_meters[loss_name].update(loss.detach().item())
                        total_loss += loss
                    total_loss = add_margin_regularization_loss(
                        total_loss, loss_meters, self.model
                    )
                    update_xbm(self.xbm, emb, targets)

                self.amp_scaler.scale(total_loss / self.grad_accum_steps).backward()

                if ((batch_index + 1) % self.grad_accum_steps == 0) or (
                    batch_index + 1 == len(train_loader)
                ):
                    self.amp_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                    self.amp_scaler.step(self.optimizer)
                    self.amp_scaler.update()
                    self._update_averaged_model()
                    if self.scheduler_step_per_batch:
                        self._step_scheduler()
                    self.optimizer.zero_grad()
            else:
                output, emb = self._forward_model(images, targets, target_extras)
                xbm = get_xbm_for_targets(self.xbm, targets)

                for loss_name, loss_params in criterion.items():
                    loss = calculate_weighted_loss(
                        loss_params,
                        output,
                        emb,
                        targets,
                        images=images,
                        xbm=xbm,
                    )
                    loss_meters[loss_name].update(loss.detach().item())
                    total_loss += loss
                total_loss = add_margin_regularization_loss(
                    total_loss, loss_meters, self.model
                )
                update_xbm(self.xbm, emb, targets)

                total_loss_grad_accum = total_loss / self.grad_accum_steps
                total_loss_grad_accum.backward()

                if ((batch_index + 1) % self.grad_accum_steps == 0) or (
                    batch_index + 1 == len(train_loader)
                ):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                    self.optimizer.step()
                    self._update_averaged_model()
                    if self.scheduler_step_per_batch:
                        self._step_scheduler()
                    self.optimizer.zero_grad()

            loss_meters["total_loss"].update(total_loss.detach().item())

            info_params = dict(
                epoch=self.epoch,
                lr=self.optimizer.param_groups[-1]["lr"],
            )
            margin_m = self._get_margin_value()
            if margin_m is not None:
                info_params["m"] = margin_m

            for loss_name, loss_meter in loss_meters.items():
                info_params[loss_name] = loss_meter.avg

            tqdm_train.set_postfix(**info_params)

        train_loss = loss_meters["total_loss"].avg
        if not self.scheduler_step_per_batch:
            self._step_scheduler(train_loss)

        stats = dict(
            losses={
                loss_name: loss_meter.avg
                for loss_name, loss_meter in loss_meters.items()
            },
        )
        margin_m = self._get_margin_value()
        if margin_m is not None:
            stats["m"] = margin_m

        return edict(stats)

    def valid_epoch(self, valid_loader):
        if hasattr(self.optimizer, "eval"):
            self.optimizer.eval()
        eval_model = self.get_eval_model()
        eval_model.eval()

        loss_meters = {k: AverageMeter() for k, v in self.loss_fns.items()}
        loss_meters["total_loss"] = AverageMeter()

        tqdm_val = tqdm(valid_loader, total=int(len(valid_loader)))

        criterion = self.loss_fns

        with torch.no_grad():
            for batch_index, (images, targets, file_names) in enumerate(tqdm_val):
                if self.debug and batch_index > 10:
                    break

                targets, target_extras = split_targets(targets)

                if self.visualize_batch:
                    if self.epoch == 1 and batch_index == 0:
                        labels = [self.ids_to_labels[anno.item()] for anno in targets]
                        save_batch_grid(
                            images.cpu(),
                            labels,
                            self.config.backbone.norm_std,
                            self.config.backbone.norm_mean,
                            save_dir=self.work_dir,
                            split="valid",
                        )

                images = images.to(self.device)
                targets = targets.to(self.device)
                target_extras = move_to_device(target_extras, self.device)

                output, emb = self._forward_model(
                    images, targets, target_extras, model=eval_model
                )

                total_loss = 0
                for loss_name, loss_params in criterion.items():
                    loss = calculate_weighted_loss(
                        loss_params,
                        output,
                        emb,
                        targets,
                        images=images,
                    )
                    loss_meters[loss_name].update(loss.detach().item())
                    total_loss += loss
                total_loss = add_margin_regularization_loss(
                    total_loss, loss_meters, self.model
                )

                loss_meters["total_loss"].update(total_loss.detach().item())

                info_params = dict(
                    epoch=self.epoch,
                )

                for loss_name, loss_meter in loss_meters.items():
                    info_params[loss_name] = loss_meter.avg

                tqdm_val.set_postfix(**info_params)

        stats = dict(
            losses={
                loss_name: loss_meter.avg
                for loss_name, loss_meter in loss_meters.items()
            },
        )

        if hasattr(self.optimizer, "train"):
            self.optimizer.train()

        return edict(stats)

    def _reset_epochs(self):
        self.epoch = 1

    def update_epoch(self):
        self.epoch += 1
