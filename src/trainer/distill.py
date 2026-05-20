from tqdm.auto import tqdm
import torch
from easydict import EasyDict as edict

from ..utils import AverageMeter, is_main_process
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


class DistillTrainer:
    def __init__(
        self,
        config,
        model,
        model_teacher,
        optimizer,
        epoch=1,
        work_dir="./",
        device=None,
        accelerator=None,
        ids_to_labels=None,
    ):

        self.config = config
        self.model = model
        self.model_teacher = model_teacher
        self.optimizer = optimizer
        self.epoch = epoch
        self.work_dir = work_dir

        self.device = device
        self.accelerator = accelerator
        self.ids_to_labels = ids_to_labels
        self.grad_accum_steps = config.train.trainer.grad_accum_steps
        self.distill_loss_only = config.distillation.trainer.params.distill_loss_only
        self.distill_loss_weight = (
            config.distillation.trainer.params.distill_loss_weight
        )

        self.n_epochs = config.epochs - epoch + 1
        set_scheduler_tmax(config.scheduler.scheduler, self.n_epochs)

        scheduler = get_scheduler(optimizer, scheduler_config=config.scheduler)
        self.scheduler_step_per_batch = scheduler_steps_per_batch(scheduler)

        if accelerator is None:
            self.distill_loss_fns = get_loss(
                loss_config=config.distillation.trainer, device=device
            )
            if not self.distill_loss_only:
                self.loss_fns = get_loss(loss_config=config.loss, device=device)
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
            self.scheduler = scheduler
            self.amp_device_type = _amp_device_type(device)
            self.amp_scaler = (
                torch.amp.GradScaler(self.amp_device_type)
                if self.amp_device_type is not None and config.amp
                else None
            )
        else:
            self.distill_loss_fns = get_loss(loss_config=config.distillation.trainer)
            if not self.distill_loss_only:
                self.loss_fns = get_loss(loss_config=config.loss)
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
            self.scheduler = accelerator.prepare(scheduler)
            self.amp_device_type = None
            self.amp_scaler = None

        self.debug = config.debug
        self.visualize_batch = config.visualize_batch
        self.dpap_transform = get_dpap_transform(config)
        self.warmup_scheduler = get_warmup_scheduler(
            self.optimizer, scheduler_config=config.scheduler
        )
        averaging_device = accelerator.device if accelerator is not None else device
        self.averaged_model = create_averaged_model(
            config,
            self._unwrap_student_model(),
            device=averaging_device,
        )
        self.xbm = create_xbm(config, device=averaging_device)

    def _get_labels_for_visualization(self, targets):
        if isinstance(targets, (list, tuple)):
            return targets[0]
        return targets

    def _unwrap_student_model(self):
        if self.accelerator is None:
            return self.model
        return self.accelerator.unwrap_model(self.model)

    def _forward_student(self, images, targets=None, target_extras=None, model=None):
        model = self.model if model is None else model
        keypoints = get_keypoints(target_extras)
        if self.distill_loss_only:
            if keypoints is not None:
                return model(images, keypoints=keypoints)
            return model(images)
        if keypoints is not None:
            return model(images, targets, keypoints=keypoints)
        return model(images, targets)

    def _apply_dpap(self, images):
        return apply_dpap_transform(self.dpap_transform, images)

    def _update_averaged_model(self):
        if self.averaged_model is None:
            return
        if should_update_averaged_model(self.config, self.epoch):
            self.averaged_model.update_parameters(self._unwrap_student_model())

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

        loss_meters = {k: AverageMeter() for k, v in self.distill_loss_fns.items()}
        if not self.distill_loss_only:
            loss_meters.update({k: AverageMeter() for k, v in self.loss_fns.items()})
        loss_meters["total_loss"] = AverageMeter()

        is_main_proc = is_main_process(self.accelerator)

        tqdm_train = tqdm(
            train_loader, total=int(len(train_loader)), disable=not is_main_proc
        )

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
                target_extras = {}
            elif self.accelerator is not None:
                target_extras = move_to_device(target_extras, self.accelerator.device)

            if self.visualize_batch and is_main_proc:
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
            if self.accelerator is not None:
                with self.accelerator.accumulate(self.model):
                    images = self._apply_dpap(images)
                    if not self.distill_loss_only:
                        output_student, emb_student = self._forward_student(
                            images, targets, target_extras
                        )
                    else:
                        emb_student = self._forward_student(
                            images, target_extras=target_extras
                        )

                    with torch.no_grad():
                        emb_teacher = self.model_teacher(images)

                    for loss_name, loss_params in self.distill_loss_fns.items():
                        loss = (
                            loss_params.loss_fn(emb_student, emb_teacher)
                            * loss_params.weight
                        )
                        if self.accelerator.is_local_main_process:
                            loss_meters[loss_name].update(loss.detach().item())
                        total_loss += loss
                    total_loss *= self.distill_loss_weight

                    if not self.distill_loss_only:
                        criterion = self.mix_loss_fns if is_mixed else self.loss_fns
                        xbm = get_xbm_for_targets(self.xbm, targets)
                        for loss_name, loss_params in criterion.items():
                            loss = calculate_weighted_loss(
                                loss_params,
                                output_student,
                                emb_student,
                                targets,
                                images=images,
                                xbm=xbm,
                            )
                            if self.accelerator.is_local_main_process:
                                loss_meters[loss_name].update(loss.detach().item())
                            total_loss += loss
                        total_loss = add_margin_regularization_loss(
                            total_loss,
                            loss_meters,
                            self.model,
                            accelerator=self.accelerator,
                            update_meter=self.accelerator.is_local_main_process,
                        )
                        update_xbm(self.xbm, emb_student, targets)

                    self.accelerator.backward(total_loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), 5)
                    self.optimizer.step()
                    if self.accelerator.sync_gradients:
                        self._update_averaged_model()
                    if (
                        self.accelerator.sync_gradients
                        and self.scheduler_step_per_batch
                    ):
                        self._step_scheduler()
                    self.optimizer.zero_grad()
            else:
                if self.amp_scaler is not None:
                    with torch.amp.autocast(self.amp_device_type):
                        images = images.to(self.device)
                        if is_mixed:
                            targets[0] = targets[0].to(self.device)
                            targets[1] = targets[1].to(self.device)
                        else:
                            targets = targets.to(self.device)
                            target_extras = move_to_device(
                                target_extras,
                                self.device,
                            )
                        images = self._apply_dpap(images)

                        if not self.distill_loss_only:
                            output_student, emb_student = self._forward_student(
                                images, targets, target_extras
                            )
                        else:
                            emb_student = self._forward_student(
                                images, target_extras=target_extras
                            )

                        with torch.no_grad():
                            emb_teacher = self.model_teacher(images)

                        for loss_name, loss_params in self.distill_loss_fns.items():
                            loss = (
                                loss_params.loss_fn(emb_student, emb_teacher)
                                * loss_params.weight
                            )
                            loss_meters[loss_name].update(loss.detach().item())
                            total_loss += loss
                        total_loss *= self.distill_loss_weight

                        if not self.distill_loss_only:
                            criterion = self.mix_loss_fns if is_mixed else self.loss_fns
                            xbm = get_xbm_for_targets(self.xbm, targets)
                            for loss_name, loss_params in criterion.items():
                                loss = calculate_weighted_loss(
                                    loss_params,
                                    output_student,
                                    emb_student,
                                    targets,
                                    images=images,
                                    xbm=xbm,
                                )
                                loss_meters[loss_name].update(loss.detach().item())
                                total_loss += loss
                            total_loss = add_margin_regularization_loss(
                                total_loss, loss_meters, self.model
                            )
                            update_xbm(self.xbm, emb_student, targets)

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
                    images = images.to(self.device)
                    if is_mixed:
                        targets[0] = targets[0].to(self.device)
                        targets[1] = targets[1].to(self.device)
                    else:
                        targets = targets.to(self.device)
                        target_extras = move_to_device(target_extras, self.device)
                    images = self._apply_dpap(images)

                    if not self.distill_loss_only:
                        output_student, emb_student = self._forward_student(
                            images, targets, target_extras
                        )
                    else:
                        emb_student = self._forward_student(
                            images, target_extras=target_extras
                        )

                    with torch.no_grad():
                        emb_teacher = self.model_teacher(images)

                    for loss_name, loss_params in self.distill_loss_fns.items():
                        loss = (
                            loss_params.loss_fn(emb_student, emb_teacher)
                            * loss_params.weight
                        )
                        loss_meters[loss_name].update(loss.detach().item())
                        total_loss += loss
                    total_loss *= self.distill_loss_weight

                    if not self.distill_loss_only:
                        criterion = self.mix_loss_fns if is_mixed else self.loss_fns
                        xbm = get_xbm_for_targets(self.xbm, targets)
                        for loss_name, loss_params in criterion.items():
                            loss = calculate_weighted_loss(
                                loss_params,
                                output_student,
                                emb_student,
                                targets,
                                images=images,
                                xbm=xbm,
                            )
                            loss_meters[loss_name].update(loss.detach().item())
                            total_loss += loss
                        total_loss = add_margin_regularization_loss(
                            total_loss, loss_meters, self.model
                        )
                        update_xbm(self.xbm, emb_student, targets)

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

            if is_main_proc:
                loss_meters["total_loss"].update(total_loss.detach().item())

                info_params = dict(
                    epoch=self.epoch,
                    lr=self.optimizer.param_groups[-1]["lr"],
                )

                for loss_name, loss_meter in loss_meters.items():
                    info_params[loss_name] = loss_meter.avg

                tqdm_train.set_postfix(**info_params)

        if not self.scheduler_step_per_batch:
            self._step_scheduler(loss_meters["total_loss"].avg)

        stats = {}
        if is_main_proc:
            stats = dict(
                losses={
                    loss_name: loss_meter.avg
                    for loss_name, loss_meter in loss_meters.items()
                },
            )

        return edict(stats)

    def valid_epoch(self, valid_loader):
        if hasattr(self.optimizer, "eval"):
            self.optimizer.eval()
        eval_model = self.get_eval_model()
        eval_model.eval()

        loss_meters = {k: AverageMeter() for k, v in self.distill_loss_fns.items()}
        if not self.distill_loss_only:
            loss_meters.update({k: AverageMeter() for k, v in self.loss_fns.items()})
        loss_meters["total_loss"] = AverageMeter()

        is_main_proc = is_main_process(self.accelerator)

        tqdm_val = tqdm(
            valid_loader, total=int(len(valid_loader)), disable=not is_main_proc
        )

        with torch.no_grad():
            for batch_index, (images, targets, file_names) in enumerate(tqdm_val):
                if self.debug and batch_index > 10:
                    break

                targets, target_extras = split_targets(targets)
                if self.accelerator is not None:
                    target_extras = move_to_device(
                        target_extras,
                        self.accelerator.device,
                    )

                if self.visualize_batch and is_main_proc:
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
                if self.accelerator is None:
                    images = images.to(self.device)
                    targets = targets.to(self.device)
                    target_extras = move_to_device(target_extras, self.device)

                if not self.distill_loss_only:
                    output_student, emb_student = self._forward_student(
                        images, targets, target_extras, model=eval_model
                    )
                else:
                    emb_student = self._forward_student(
                        images, target_extras=target_extras, model=eval_model
                    )

                emb_teacher = self.model_teacher(images)

                total_loss = 0
                for loss_name, loss_params in self.distill_loss_fns.items():
                    loss = (
                        loss_params.loss_fn(emb_student, emb_teacher)
                        * loss_params.weight
                    )
                    if is_main_proc:
                        loss_meters[loss_name].update(loss.detach().item())
                    total_loss += loss
                total_loss *= self.distill_loss_weight

                if not self.distill_loss_only:
                    for loss_name, loss_params in self.loss_fns.items():
                        loss = calculate_weighted_loss(
                            loss_params,
                            output_student,
                            emb_student,
                            targets,
                            images=images,
                        )
                        loss_meters[loss_name].update(loss.detach().item())
                        total_loss += loss
                    total_loss = add_margin_regularization_loss(
                        total_loss,
                        loss_meters,
                        self.model,
                        accelerator=self.accelerator,
                        update_meter=is_main_proc,
                    )

                if is_main_proc:
                    loss_meters["total_loss"].update(total_loss.detach().item())

                    info_params = dict(
                        epoch=self.epoch,
                    )

                    for loss_name, loss_meter in loss_meters.items():
                        info_params[loss_name] = loss_meter.avg

                    tqdm_val.set_postfix(**info_params)

        stats = {}
        if is_main_proc:
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
