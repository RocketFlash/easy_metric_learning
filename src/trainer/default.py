from tqdm.auto import tqdm
import torch
from torch.cuda import amp
from easydict import EasyDict as edict

from ..utils import AverageMeter
from ..model.margin.utils import get_incremental_margin
from ..transform import mix_transform
from ..loss.mix import MixCriterion
from ..loss import get_loss
from ..visualization import save_batch_grid
from ..scheduler import (get_scheduler,
                         get_warmup_scheduler)


class MLTrainer:
    def __init__(
            self, 
            config,
            model, 
            optimizer,
            epoch=1,
            work_dir='./', 
            device='cpu', 
            ids_to_labels=None
        ):

        self.config = config
        self.model  = model
        self.epoch  = epoch
        self.device = device
        self.work_dir = work_dir
        self.optimizer = optimizer
        self.ids_to_labels = ids_to_labels

        self.n_epochs   = config.epochs - epoch + 1
        self.loss_fns   = get_loss(loss_config=config.loss, device=device)
        self.amp_scaler = amp.GradScaler() if device != 'cpu' and config.amp else None
        self.scheduler  = get_scheduler(self.optimizer, scheduler_config=config.scheduler)
        self.warmup_scheduler = get_warmup_scheduler(self.optimizer, scheduler_config=config.scheduler)
        
        self.debug = config.debug
        self.visualize_batch = config.visualize_batch
        self.grad_accum_steps = config.train.grad_accum_steps

        self.mix_loss_fns = {k: MixCriterion(v) for k, v in self.loss_fns.items()}
        
        if config.train.incremental_margin is not None:
            self.incremental_margin = get_incremental_margin(
                m_max=self.model.margin.m,
                m_min=config.train.incremental_margin.min_m,
                n_epochs=self.n_epochs,
                mode=config.train.incremental_margin.type
            )
        else:
            self.incremental_margin = None


    def train_epoch(self, train_loader):
        self.model.train()

        if self.incremental_margin is not None:
            self.model.margin.update(self.incremental_margin[self.epoch-1])

        loss_meters = {k: AverageMeter() for k, v in self.loss_fns.items()}
        loss_meters['total_loss'] = AverageMeter()
        
        tqdm_train = tqdm(
            train_loader, 
            total=int(len(train_loader))
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
            
            if is_mixed:
                criterion = self.mix_loss_fns
                targets[0] = targets[0].to(self.device)
                targets[1] = targets[1].to(self.device)
            else:
                criterion = self.loss_fns
                targets = targets.to(self.device)

            images = images.to(self.device)
            
            if self.visualize_batch:
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
            if self.amp_scaler is not None:
                with amp.autocast():
                    output = self.model(images, targets)

                    for loss_name, loss_params in criterion.items():
                        loss = loss_params.loss_fn(output, targets) * loss_params.weight
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
                output = self.model(images, targets)
                
                for loss_name, loss_params in criterion.items():
                    loss = loss_params.loss_fn(output, targets) * loss_params.weight
                    loss_meters[loss_name].update(loss.detach().item())
                    total_loss += loss

                total_loss_grad_accum = total_loss / self.grad_accum_steps
                total_loss_grad_accum.backward()

                if ((batch_index + 1) % self.grad_accum_steps == 0) or (batch_index + 1 == len(train_loader)):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            loss_meters['total_loss'].update(total_loss.detach().item())

            if self.warmup_scheduler is not None:
                if batch_index < len(tqdm_train)-1:
                    with self.warmup_scheduler.dampening(): pass

            info_params = dict(
                epoch=self.epoch, 
                lr=self.optimizer.param_groups[-1]['lr'],
                m=self.model.margin.m
            )

            for loss_name, loss_meter in loss_meters.items():
                info_params[loss_name] = loss_meter.avg

            tqdm_train.set_postfix(**info_params)

        if self.warmup_scheduler is not None:
            with self.warmup_scheduler.dampening():
                self.scheduler.step()
        else:
            self.scheduler.step()

        stats = dict(
            losses={loss_name: loss_meter.avg for loss_name, loss_meter in loss_meters.items()},
        )

        return edict(stats)
    

    def valid_epoch(self, valid_loader):
        self.model.eval()

        loss_meters = {k: AverageMeter() for k, v in self.loss_fns.items()}
        loss_meters['total_loss'] = AverageMeter()

        tqdm_val = tqdm(valid_loader, total=int(len(valid_loader)))

        criterion = self.loss_fns
        
        with torch.no_grad():
            for batch_index, (images, targets, file_names) in enumerate(tqdm_val):
                if self.debug and batch_index>10: break

                if self.visualize_batch:
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

                images = images.to(self.device)
                targets = targets.to(self.device)
                
                output = self.model(images, targets)

                total_loss = 0
                for loss_name, loss_params in criterion.items():
                    loss = loss_params.loss_fn(output, targets) * loss_params.weight
                    loss_meters[loss_name].update(loss.detach().item())
                    total_loss += loss

                loss_meters['total_loss'].update(total_loss.detach().item())
                
                info_params = dict(
                    epoch=self.epoch, 
                )

                for loss_name, loss_meter in loss_meters.items():
                    info_params[loss_name] = loss_meter.avg
                
                tqdm_val.set_postfix(**info_params)
        
        stats = dict(
            losses={loss_name: loss_meter.avg for loss_name, loss_meter in loss_meters.items()},
        )

        return edict(stats)


    def _reset_epochs(self):
        self.epoch = 1

    
    def update_epoch(self):
        self.epoch += 1
