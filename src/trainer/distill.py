from tqdm.auto import tqdm
import torch
from torch.cuda import amp
from easydict import EasyDict as edict

from ..utils import (AverageMeter,
                     is_main_process)
from ..transform import mix_transform
from ..loss import get_loss
from ..visualization import save_batch_grid
from ..scheduler import get_scheduler


class DistillTrainer:
    def __init__(
            self, 
            config,
            model, 
            model_teacher,
            optimizer,
            epoch=1,
            work_dir='./', 
            device=None,
            accelerator=None, 
            ids_to_labels=None
        ):

        self.config = config
        self.model  = model
        self.model_teacher = model_teacher
        self.optimizer = optimizer
        self.epoch  = epoch
        self.work_dir = work_dir
        
        self.device = device
        self.accelerator = accelerator
        self.ids_to_labels = ids_to_labels
        self.grad_accum_steps = config.train.trainer.grad_accum_steps
        self.distill_loss_only = config.distillation.trainer.params.distill_loss_only
        self.distill_loss_weight = config.distillation.trainer.params.distill_loss_weight

        self.n_epochs = config.epochs - epoch + 1
        if 'T_max' in config.scheduler.scheduler:
            config.scheduler.scheduler.T_max = self.n_epochs - 1

        scheduler = get_scheduler(optimizer, scheduler_config=config.scheduler)

        if accelerator is None:
            self.distill_loss_fns = get_loss(
                loss_config=config.distillation.trainer,
                device=device
            )
            if not self.distill_loss_only:
                self.loss_fns = get_loss(loss_config=config.loss, device=device)
            self.scheduler = scheduler
            self.amp_scaler = amp.GradScaler() if device != 'cpu' and config.amp else None
        else:
            self.distill_loss_fns = get_loss(loss_config=config.distillation.trainer)
            if not self.distill_loss_only:
                self.loss_fns = get_loss(loss_config=config.loss)
            self.scheduler = accelerator.prepare(scheduler)
            self.amp_scaler = None

        self.debug = config.debug
        self.visualize_batch = config.visualize_batch
        

    def train_epoch(self, train_loader):
        self.model.train()

        loss_meters = {k: AverageMeter() for k, v in self.distill_loss_fns.items()}
        if not self.distill_loss_only:
            loss_meters.update({k: AverageMeter() for k, v in self.loss_fns.items()})
        loss_meters['total_loss'] = AverageMeter()
        
        is_main_proc = is_main_process(self.accelerator)
            
        tqdm_train = tqdm(
            train_loader, 
            total=int(len(train_loader)),
            disable=not is_main_proc
        )
        
        for batch_index, (images, targets, file_names) in enumerate(tqdm_train):
            if self.debug and batch_index>=10: break

            images, targets, is_mixed = mix_transform(
                images, 
                targets,
                cutmix_p=self.config.transform.cutmix.p,
                cutmix_alpha=self.config.transform.cutmix.alpha,
                mixup_p=self.config.transform.mixup.p,
                mixup_alpha=self.config.transform.mixup.alpha
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
                        split='train', 
                    )

            total_loss = 0
            if self.accelerator is not None:
                with self.accelerator.accumulate(self.model):
                    if not self.distill_loss_only:
                        output_student, emb_student = self.model(images, targets)
                    else:
                        emb_student = self.model(images)

                    with torch.no_grad():
                        emb_teacher = self.model_teacher(images)

                    for loss_name, loss_params in self.distill_loss_fns.items():
                        loss = loss_params.loss_fn(emb_student, emb_teacher) * loss_params.weight
                        if self.accelerator.is_local_main_process:
                            loss_meters[loss_name].update(loss.detach().item())
                        total_loss += loss
                    total_loss *= self.distill_loss_weight

                    if not self.distill_loss_only:
                        for loss_name, loss_params in self.loss_fns.items():
                            loss = loss_params.loss_fn(output_student, targets) * loss_params.weight
                            if self.accelerator.is_local_main_process:
                                loss_meters[loss_name].update(loss.detach().item())
                            total_loss += loss

                    self.accelerator.backward(total_loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), 5)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            else:
                if self.amp_scaler is not None:
                    with amp.autocast():
                        images = images.to(self.device)
                        targets = targets.to(self.device)

                        if not self.distill_loss_only:
                            output_student, emb_student = self.model(images, targets)
                        else:
                            emb_student = self.model(images)

                        with torch.no_grad():
                            emb_teacher = self.model_teacher(images)

                        for loss_name, loss_params in self.distill_loss_fns.items():
                            loss = loss_params.loss_fn(emb_student, emb_teacher) * loss_params.weight
                            loss_meters[loss_name].update(loss.detach().item())
                            total_loss += loss
                        total_loss *= self.distill_loss_weight

                        if not self.distill_loss_only:
                            for loss_name, loss_params in self.loss_fns.items():
                                loss = loss_params.loss_fn(output_student, targets) * loss_params.weight
                                loss_meters[loss_name].update(loss.detach().item())
                                total_loss += loss

                    self.amp_scaler.scale(total_loss / self.grad_accum_steps).backward()

                    if ((batch_index + 1) % self.grad_accum_steps == 0) or (batch_index + 1 == len(train_loader)):
                        self.amp_scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                        self.amp_scaler.step(self.optimizer)
                        self.amp_scaler.update()
                        self.optimizer.zero_grad()
                else:
                    images = images.to(self.device)
                    targets = targets.to(self.device)

                    if not self.distill_loss_only:
                        output_student, emb_student = self.model(images, targets)
                    else:
                        emb_student = self.model(images)

                    with torch.no_grad():
                        emb_teacher = self.model_teacher(images)
                    
                    for loss_name, loss_params in self.distill_loss_fns.items():
                        loss = loss_params.loss_fn(emb_student, emb_teacher) * loss_params.weight
                        loss_meters[loss_name].update(loss.detach().item())
                        total_loss += loss
                    total_loss *= self.distill_loss_weight

                    if not self.distill_loss_only:
                        for loss_name, loss_params in self.loss_fns.items():
                            loss = loss_params.loss_fn(output_student, targets) * loss_params.weight
                            loss_meters[loss_name].update(loss.detach().item())
                            total_loss += loss

                    total_loss_grad_accum = total_loss / self.grad_accum_steps
                    total_loss_grad_accum.backward()

                    if ((batch_index + 1) % self.grad_accum_steps == 0) or (batch_index + 1 == len(train_loader)):
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
            
            if is_main_proc:
                loss_meters['total_loss'].update(total_loss.detach().item())

                info_params = dict(
                    epoch=self.epoch, 
                    lr=self.optimizer.param_groups[-1]['lr'],
                )

                for loss_name, loss_meter in loss_meters.items():
                    info_params[loss_name] = loss_meter.avg

                tqdm_train.set_postfix(**info_params)

        self.scheduler.step()

        stats = {}
        if is_main_proc:
            stats = dict(
                losses={loss_name: loss_meter.avg for loss_name, loss_meter in loss_meters.items()},
            )

        return edict(stats)
    

    def valid_epoch(self, valid_loader):
        self.model.eval()

        loss_meters = {k: AverageMeter() for k, v in self.distill_loss_fns.items()}
        if not self.distill_loss_only:
            loss_meters.update({k: AverageMeter() for k, v in self.loss_fns.items()})
        loss_meters['total_loss'] = AverageMeter()

        is_main_proc = is_main_process(self.accelerator)

        tqdm_val = tqdm(
            valid_loader, 
            total=int(len(valid_loader)),
            disable=not is_main_proc
        )
        
        with torch.no_grad():
            for batch_index, (images, targets, file_names) in enumerate(tqdm_val):
                if self.debug and batch_index>10: break

                if self.visualize_batch and is_main_proc:
                    if self.epoch == 1 and batch_index == 0:
                        labels = [self.ids_to_labels[anno.item()] for anno in targets]
                        save_batch_grid(
                            images.cpu(), 
                            labels,
                            self.config.backbone.norm_std,
                            self.config.backbone.norm_mean,
                            save_dir=self.work_dir, 
                            split='valid', 
                        )
                if not self.distill_loss_only:
                    output_student, emb_student = self.model(images)
                else:
                    emb_student = self.model(images)

                emb_teacher = self.model_teacher(images)

                total_loss = 0
                for loss_name, loss_params in self.loss_fns.items():
                    loss = loss_params.loss_fn(emb_student, emb_teacher) * loss_params.weight
                    if is_main_proc:
                        loss_meters[loss_name].update(loss.detach().item())
                    total_loss += loss
                total_loss *= self.distill_loss_weight

                if not self.distill_loss_only:
                    for loss_name, loss_params in self.loss_fns.items():
                        loss = loss_params.loss_fn(output_student, targets) * loss_params.weight
                        loss_meters[loss_name].update(loss.detach().item())
                        total_loss += loss

                if is_main_proc:
                    loss_meters['total_loss'].update(total_loss.detach().item())
                
                    info_params = dict(
                        epoch=self.epoch, 
                    )

                    for loss_name, loss_meter in loss_meters.items():
                        info_params[loss_name] = loss_meter.avg
                    
                    tqdm_val.set_postfix(**info_params)
        
        stats = {}
        if is_main_proc:
            stats = dict(
                losses={loss_name: loss_meter.avg for loss_name, loss_meter in loss_meters.items()},
            )

        return edict(stats)


    def _reset_epochs(self):
        self.epoch = 1

    
    def update_epoch(self):
        self.epoch += 1
