import os
from tqdm.auto import tqdm
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch.cuda import amp
from easydict import EasyDict as edict

from .utils import AverageMeter
from .utils import batch_grid
from .metric import accuracy, GAP
from .margin.utils import get_incremental_margin
from .transform import cutmix, mixup
from .loss import MixCriterion


class DistillationTrainer:
    def __init__(
            self, 
            model, 
            model_teacher,
            optimizer, 
            logger, 
            work_dir='./', 
            device='cpu', 
            epoch=1,
            n_epochs=10,
            grad_accum_steps=1, 
            amp_scaler=None, 
            warmup_scheduler=None, 
            is_debug=False,
            visualize_batch=False,
            p_mixup=0.0,
            p_cutmix=0.0,
        ):
        self.model = model
        self.model_teacher = model_teacher
        self.optimizer = optimizer
        self.logger = logger
        self.epoch = epoch
        self.grad_accum_steps = grad_accum_steps
        self.work_dir = work_dir
        self.device = device
        self.is_debug = is_debug
        self.amp_scaler = amp_scaler
        self.warmup_scheduler = warmup_scheduler
        self.n_epochs = n_epochs
        self.visualize_batch = visualize_batch
        self.loss_distill = nn.MSELoss()
        self.p_mixup  = p_mixup
        self.p_cutmix = p_cutmix
        self.alpha_mixup  = 0.2
        self.alpha_cutmix = 1
        

    def train_epoch(self, train_loader):
        self.model.train()

        meter_loss_total  = AverageMeter()
        
        tqdm_train = tqdm(train_loader, 
                          total=int(len(train_loader)))
        images_wdb = []

        for batch_index, (data, targets) in enumerate(tqdm_train):
            if self.is_debug and batch_index>=10: break

            p = np.random.rand()
            if self.p_cutmix>0 or self.p_mixup>0:
                if p < self.p_cutmix and p < self.p_mixup:
                    p = np.random.rand()
                    if p < 0.5:
                        data, targets = cutmix(data, targets, self.alpha_cutmix)
                    else:
                        data, targets = mixup(data, targets, self.alpha_mixup)
                elif p < self.p_cutmix:
                    data, targets = cutmix(data, targets, self.alpha_cutmix)
                elif p < self.p_mixup:
                    data, targets = mixup(data, targets, self.alpha_mixup)

            data = data.to(self.device)
            
            if self.visualize_batch:
                if not self.is_debug:
                    if self.epoch == 1 and batch_index == 0:
                        image_grid = batch_grid(data)
                        save_path = os.path.join(self.work_dir, 
                                                f'train_batch_{batch_index}.png')
                        torchvision.utils.save_image(image_grid, save_path)
                        images_wdb.append(wandb.Image(save_path, 
                                                    caption=f'train_batch_{batch_index}'))

            if self.amp_scaler is not None:
                with amp.autocast():
                    output_student = self.model(data)
                    output_teacher = self.model_teacher(data)
                    
                    loss = self.loss_distill(output_student, output_teacher)
                    loss /= self.grad_accum_steps

                self.amp_scaler.scale(loss).backward()

                if ((batch_index + 1) % self.grad_accum_steps == 0) or (batch_index + 1 == len(train_loader)):
                    self.amp_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                    self.amp_scaler.step(self.optimizer)
                    self.amp_scaler.update()
                    self.optimizer.zero_grad()
            else:
                output_student = self.model(data)
                output_teacher = self.model_teacher(data)

                loss = self.loss_distill(output_student, output_teacher)
                loss /= self.grad_accum_steps
                loss.backward()

                if ((batch_index + 1) % self.grad_accum_steps == 0) or (batch_index + 1 == len(train_loader)):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            if self.warmup_scheduler is not None:
                if batch_index < len(tqdm_train)-1:
                    with self.warmup_scheduler.dampening(): pass
            
            meter_loss_total.update(loss.detach().item())
            info_params = dict(
                epoch=self.epoch, 
                loss=meter_loss_total.avg ,
                lr=self.optimizer.param_groups[-1]['lr']
            )

            tqdm_train.set_postfix(**info_params)

        stats = dict(
            loss=meter_loss_total.avg,
            images_wdb=images_wdb,
        )

        return edict(stats)
    

    def valid_epoch(self, valid_loader):
        self.model.eval()

        meter_loss_total  = AverageMeter()
        tqdm_val = tqdm(valid_loader, total=int(len(valid_loader)))
        images_wdb = []

        with torch.no_grad():
            for batch_index, (data, targets) in enumerate(tqdm_val):
                if self.is_debug and batch_index>10: break

                if self.visualize_batch:
                    if not self.is_debug:
                        if self.epoch == 1 and batch_index == 0:
                            image_grid = batch_grid(data)
                            save_path = os.path.join(self.work_dir, 
                                                    f'valid_batch_{batch_index}.png')
                            torchvision.utils.save_image(image_grid, save_path)
                            images_wdb.append(wandb.Image(save_path, 
                                                        caption=f'valid_batch_{batch_index}'))

                data = data.to(self.device)
                targets = targets.to(self.device)
                
                output_student = self.model(data)
                output_teacher = self.model_teacher(data)
                loss = self.loss_distill(output_student, output_teacher)
                
                meter_loss_total.update(loss.detach().item())
                
                info_params = dict(
                    epoch=self.epoch, 
                    loss=meter_loss_total.avg 
                )

                tqdm_val.set_postfix(**info_params)
        
        stats = dict(
            loss=meter_loss_total.avg,
            images_wdb=images_wdb,
            f1_cat=None,
            loss_cat=None,
            loss_margin=None
        )

        return edict(stats)


    def _reset_epochs(self):
        self.epoch = 1

    
    def update_epoch(self):
        self.epoch += 1