import os
from tqdm.auto import tqdm
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch.cuda import amp
from easydict import EasyDict as edict

from ..utils import AverageMeter
from ..metric import GAP
from ..model.margin.utils import get_incremental_margin
from ..transform import cutmix, mixup
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
            logger, 
            epoch=1,
            work_dir='./', 
            device='cpu', 
            ids_to_labels=None
        ):

        self.config = config
        self.model  = model
        self.logger = logger
        self.epoch  = epoch
        self.device = device
        self.work_dir = work_dir
        self.optimizer = optimizer
        self.ids_to_labels = ids_to_labels

        self.n_epochs   = config.epochs - epoch + 1
        self.loss_func  = get_loss(loss_config=config.loss).to(device)
        self.amp_scaler = amp.GradScaler() if device != 'cpu' and config.amp else None
        self.scheduler  = get_scheduler(self.optimizer, scheduler_config=config.scheduler)
        self.warmup_scheduler = get_warmup_scheduler(self.optimizer, scheduler_config=config.scheduler)
        
        self.debug = config.debug
        self.visualize_batch = config.visualize_batch
        self.calculate_GAP = config.train.calculate_GAP
        self.grad_accum_steps = config.train.grad_accum_steps

        self.mix_loss = MixCriterion(self.loss_func)
        
        if config.train.incremental_margin is not None:
            self.incremental_margin = get_incremental_margin(
                m_max=self.model.margin.m,
                m_min=config.train.incremental_margin.min_m,
                n_epochs=self.n_epochs,
                mode=config.train.incremental_margin.type
            )
        else:
            self.incremental_margin = None


    def mix_transform(self, images, targets):
        p = np.random.rand()
        is_mixed = False

        if self.config.train.cutmix.p>0 or self.config.train.mixup.p>0:
            if p < self.config.train.cutmix.p and p < self.config.train.mixup.p:
                p = np.random.rand()
                if p < 0.5:
                    images, targets = cutmix(images, targets, self.config.train.cutmix.alpha)
                else:
                    images, targets = mixup(images, targets, self.config.train.mixup.alpha)
                is_mixed = True
            elif p < self.p_cutmix:
                images, targets = cutmix(images, targets, self.config.train.cutmix.alpha)
                is_mixed = True
            elif p < self.p_mixup:
                images, targets = mixup(images, targets, self.config.train.mixup.alpha)
                is_mixed = True

        return images, targets, is_mixed


    def train_epoch(self, train_loader):
        self.model.train()

        if self.incremental_margin is not None:
            self.model.margin.update(self.incremental_margin[self.epoch-1])

        meter_loss = AverageMeter()
        
        tqdm_train = tqdm(train_loader, 
                          total=int(len(train_loader)))
        
        for batch_index, (images, targets) in enumerate(tqdm_train):
            if self.debug and batch_index>=10: break

            images, targets, is_mixed = self.mix_transform(images, targets)
            
            if is_mixed:
                criterion = self.mix_loss
                targets[0] = targets[0].to(self.device)
                targets[1] = targets[1].to(self.device)
            else:
                criterion = self.loss_func
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
                        batch_index=batch_index
                    )

            if self.amp_scaler is not None:
                with amp.autocast():
                    output = self.model(images, targets)
                    loss = criterion(output, targets)
                    loss /= self.grad_accum_steps
                self.amp_scaler.scale(loss).backward()

                if ((batch_index + 1) % self.grad_accum_steps == 0) or (batch_index + 1 == len(train_loader)):
                    self.amp_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                    self.amp_scaler.step(self.optimizer)
                    self.amp_scaler.update()
                    self.optimizer.zero_grad()
            else:
                output = self.model(images, targets)
                
                loss = criterion(output, targets)
                loss /= self.grad_accum_steps
                loss.backward()

                if ((batch_index + 1) % self.grad_accum_steps == 0) or (batch_index + 1 == len(train_loader)):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            meter_loss.update(loss.detach().item())

            if self.warmup_scheduler is not None:
                if batch_index < len(tqdm_train)-1:
                    with self.warmup_scheduler.dampening(): pass

            info_params = dict(
                epoch=self.epoch, 
                loss=meter_loss.avg,
                lr=self.optimizer.param_groups[-1]['lr'],
                m=self.model.margin.m
            )

            tqdm_train.set_postfix(**info_params)

        if self.warmup_scheduler is not None:
            with self.warmup_scheduler.dampening():
                self.scheduler.step()
        else:
            self.scheduler.step()

        stats = dict(
            loss=meter_loss.avg,
        )

        return edict(stats)
    

    def valid_epoch(self, valid_loader):
        self.model.eval()

        meter_loss = AverageMeter()
        activation = nn.Softmax(dim=1)

        tqdm_val = tqdm(valid_loader, total=int(len(valid_loader)))
        vals_gt, vals_pred, vals_conf = [], [], []
        
        with torch.no_grad():
            for batch_index, (images, targets) in enumerate(tqdm_val):
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
                            batch_index=batch_index
                        )

                images = images.to(self.device)
                targets = targets.to(self.device)
                
                output = self.model(images, targets)
                loss = self.loss_func(output, targets)

                if self.calculate_GAP:
                    output_probs = activation(output)
                    confs, pred = torch.max(output_probs, dim=1)

                    vals_conf.extend(confs.cpu().numpy().tolist())
                    vals_pred.extend(pred.cpu().numpy().tolist())
                    vals_gt.extend(targets.cpu().numpy().tolist())

                meter_loss.update(loss.detach().item())
                
                info_params = dict(
                    epoch=self.epoch, 
                    loss=meter_loss.avg,
                )
                
                tqdm_val.set_postfix(**info_params)
        
        stats = dict(
            loss=meter_loss.avg,
        )

        if self.calculate_GAP:
            gap_val = GAP(vals_pred, vals_conf, vals_gt)
            stats['gap'] = gap_val

        return edict(stats)


    def _reset_epochs(self):
        self.epoch = 1

    
    def update_epoch(self):
        self.epoch += 1
