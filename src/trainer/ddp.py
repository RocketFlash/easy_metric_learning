from tqdm.auto import tqdm
import torch
from easydict import EasyDict as edict

from ..utils import AverageMeter
from ..model.margin.utils import get_incremental_margin
from ..transform import mix_transform
from ..loss.mix import MixCriterion
from ..loss import get_loss
from ..visualization import save_batch_grid
from ..scheduler import get_scheduler


class DDPTrainer:
    def __init__(
            self, 
            config,
            model, 
            optimizer,
            epoch=1,
            work_dir='./', 
            accelerator=None, 
            ids_to_labels=None
        ):

        self.config = config
        self.model  = model
        self.optimizer = optimizer
        self.epoch  = epoch
        self.work_dir = work_dir
        
        self.accelerator = accelerator
        self.ids_to_labels = ids_to_labels

        self.n_epochs = config.epochs - epoch + 1
        self.loss_fns = get_loss(loss_config=config.loss)
        if 'T_max' in config.scheduler.scheduler:
            config.scheduler.scheduler.T_max = self.n_epochs - 1
        scheduler = get_scheduler(optimizer, scheduler_config=config.scheduler)

        self.scheduler = accelerator.prepare(scheduler)
           
        self.debug = config.debug
        self.visualize_batch = config.visualize_batch
        
        self.mix_loss_fns = {k: MixCriterion(v) for k, v in self.loss_fns.items()}
        
        if config.margin.incremental_margin is not None:
            self.incremental_margin = get_incremental_margin(
                m_max=self.accelerator.unwrap_model(self.model).margin.m,
                m_min=config.margin.incremental_margin.min_m,
                n_epochs=config.epochs,
                mode=config.margin.incremental_margin.type
            )
        else:
            self.incremental_margin = None


    def train_epoch(self, train_loader):
        self.model.train()

        if self.incremental_margin is not None:
            self.accelerator.unwrap_model(self.model).margin.update(self.incremental_margin[self.epoch-1])

        loss_meters = {k: AverageMeter() for k, v in self.loss_fns.items()}
        loss_meters['total_loss'] = AverageMeter()
        
        tqdm_train = tqdm(
            train_loader, 
            total=int(len(train_loader)),
            disable=not self.accelerator.is_local_main_process
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
                targets[0] = targets[0]
                targets[1] = targets[1]
            else:
                criterion = self.loss_fns
                targets = targets
            images = images
            
            if self.visualize_batch and self.accelerator.is_local_main_process:
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
            with self.accelerator.accumulate(self.model):
                output = self.model(images, targets)
                for loss_name, loss_params in criterion.items():
                    loss = loss_params.loss_fn(output, targets) * loss_params.weight
                    if self.accelerator.is_local_main_process:
                        loss_meters[loss_name].update(loss.detach().item())
                    total_loss += loss
                self.accelerator.backward(total_loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 5)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            if self.accelerator.is_local_main_process:
                loss_meters['total_loss'].update(total_loss.detach().item())

                info_params = dict(
                    epoch=self.epoch, 
                    lr=self.optimizer.param_groups[-1]['lr'],
                    m=self.accelerator.unwrap_model(self.model).margin.m
                )

                for loss_name, loss_meter in loss_meters.items():
                    info_params[loss_name] = loss_meter.avg

                tqdm_train.set_postfix(**info_params)

        self.scheduler.step()

        stats = {}
        if self.accelerator.is_local_main_process:
            stats = dict(
                m=self.accelerator.unwrap_model(self.model).margin.m,
                losses={loss_name: loss_meter.avg for loss_name, loss_meter in loss_meters.items()},
            )

        return edict(stats)
    

    def valid_epoch(self, valid_loader):
        self.model.eval()

        loss_meters = {k: AverageMeter() for k, v in self.loss_fns.items()}
        loss_meters['total_loss'] = AverageMeter()

        tqdm_val = tqdm(
            valid_loader, 
            total=int(len(valid_loader)),
            disable=not self.accelerator.is_local_main_process
        )

        criterion = self.loss_fns
        
        with torch.no_grad():
            for batch_index, (images, targets, file_names) in enumerate(tqdm_val):
                if self.debug and batch_index>10: break

                if self.visualize_batch and self.accelerator.is_local_main_process:
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
                
                output = self.model(images, targets)

                total_loss = 0
                for loss_name, loss_params in criterion.items():
                    loss = loss_params.loss_fn(output, targets) * loss_params.weight
                    if self.accelerator.is_local_main_process:
                        loss_meters[loss_name].update(loss.detach().item())
                    total_loss += loss

                if self.accelerator.is_local_main_process:
                    loss_meters['total_loss'].update(total_loss.detach().item())
                
                    info_params = dict(
                        epoch=self.epoch, 
                    )

                    for loss_name, loss_meter in loss_meters.items():
                        info_params[loss_name] = loss_meter.avg
                    
                    tqdm_val.set_postfix(**info_params)
        
        stats = {}
        if self.accelerator.is_local_main_process:
            stats = dict(
                losses={loss_name: loss_meter.avg for loss_name, loss_meter in loss_meters.items()},
            )

        return edict(stats)


    def _reset_epochs(self):
        self.epoch = 1

    
    def update_epoch(self):
        self.epoch += 1
