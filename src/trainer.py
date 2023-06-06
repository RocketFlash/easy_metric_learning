import os
from tqdm.auto import tqdm
import torch
import torchvision
import torch.nn as nn
from torch.cuda import amp


from .utils import AverageMeter
from .utils import batch_grid
from .metric import accuracy, GAP
from .margin.utils import get_incremental_margin

try:
    import wandb
except ModuleNotFoundError:
    print('wandb is not installed')


class MLTrainer:
    def __init__(self, 
                 model, 
                 optimizer, 
                 loss_func, 
                 logger, 
                 work_dir='./', 
                 device='cpu', 
                 epoch=1,
                 n_epochs=10,
                 grad_accum_steps=1, 
                 amp_scaler=None, 
                 warmup_scheduler=None, 
                 wandb_available=True, 
                 is_debug=False,
                 calculate_GAP=True,
                 incremental_margin=None):
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.logger = logger
        self.epoch = epoch
        self.grad_accum_steps = grad_accum_steps
        self.wandb_available = wandb_available
        self.work_dir = work_dir
        self.device = device
        self.is_debug = is_debug
        self.amp_scaler = amp_scaler
        self.warmup_scheduler = warmup_scheduler
        self.calculate_GAP = calculate_GAP
        self.n_epochs = n_epochs
        
        if incremental_margin is not None:
            m_min  = incremental_margin['MIN_M']
            m_mode = incremental_margin['TYPE']
            self.incremental_margin = get_incremental_margin(m_max=self.model.margin.m,
                                                             m_min=m_min,
                                                             n_epochs=n_epochs,
                                                             mode=m_mode)
        else:
            self.incremental_margin = None

    def train_epoch(self, train_loader):
        self.model.train()

        if self.incremental_margin is not None:
            self.model.margin.update(self.incremental_margin[self.epoch-1])

        train_loss = AverageMeter()
        train_acc = AverageMeter()
        
        tqdm_train = tqdm(train_loader, total=int(len(train_loader)))
        images_wdb = []

        for batch_index, (data, targets) in enumerate(tqdm_train):
            if self.is_debug and batch_index>=10: break

            if not self.is_debug and self.wandb_available:
                if self.epoch == 1 and batch_index == 0:
                    image_grid = batch_grid(data)
                    save_path = os.path.join(self.work_dir, 
                                             f'train_batch_{batch_index}.png')
                    torchvision.utils.save_image(image_grid, save_path)
                    images_wdb.append(wandb.Image(save_path, 
                                                  caption=f'train_batch_{batch_index}'))

            data = data.to(self.device)
            targets = targets.to(self.device)

            if self.amp_scaler is not None:
                with amp.autocast():
                    output = self.model(data, targets)
                    loss = self.loss_func(output, targets)
                    acc = accuracy(output, targets)
                loss /= self.grad_accum_steps
                self.amp_scaler.scale(loss).backward()

                if ((batch_index + 1) % self.grad_accum_steps == 0) or (batch_index + 1 == len(train_loader)):
                    self.amp_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                    self.amp_scaler.step(self.optimizer)
                    self.amp_scaler.update()
                    self.optimizer.zero_grad()
            else:
                output = self.model(data, targets)
                acc = accuracy(output, targets)
                loss = self.loss_func(output, targets)
                loss /= self.grad_accum_steps
                loss.backward()

                if ((batch_index + 1) % self.grad_accum_steps == 0) or (batch_index + 1 == len(train_loader)):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            train_loss.update(loss.detach().item())

            if self.warmup_scheduler is not None:
                if batch_index < len(tqdm_train)-1:
                    with self.warmup_scheduler.dampening(): pass

            
            train_acc.update(acc)

            tqdm_train.set_postfix(epoch=self.epoch, 
                                   train_loss=train_loss.avg ,
                                   train_acc=train_acc.avg,
                                   lr=self.optimizer.param_groups[-1]['lr'],
                                   m=self.model.margin.m)

        return train_loss.avg, train_acc.avg, images_wdb
    

    def valid_epoch(self, valid_loader):
        self.model.eval()

        valid_loss = AverageMeter()
        valid_acc = AverageMeter()

        activation = nn.Softmax(dim=1)

        tqdm_val = tqdm(valid_loader, total=int(len(valid_loader)))
        vals_gt, vals_pred, vals_conf = [], [], []
        images_wdb = []

        with torch.no_grad():
            for batch_index, (data, targets) in enumerate(tqdm_val):
                batch_size = data.size(0)
                if self.is_debug and batch_index>10: break

                if not self.is_debug and self.wandb_available:
                    if self.epoch == 1 and batch_index == 0:
                        image_grid = batch_grid(data)
                        save_path = os.path.join(self.work_dir, 
                                                 f'valid_batch_{batch_index}.png')
                        torchvision.utils.save_image(image_grid, save_path)
                        images_wdb.append(wandb.Image(save_path, 
                                                      caption=f'valid_batch_{batch_index}'))

                data = data.to(self.device)
                targets = targets.to(self.device)
                
                output = self.model(data, targets)

                if self.calculate_GAP:
                    output_probs = activation(output)
                    confs, pred = torch.max(output_probs, dim=1)

                    vals_conf.extend(confs.cpu().numpy().tolist())
                    vals_pred.extend(pred.cpu().numpy().tolist())
                    vals_gt.extend(targets.cpu().numpy().tolist())

                loss = self.loss_func(output, targets)
                acc = accuracy(output, targets)

                valid_loss.update(loss.detach().item(), batch_size)
                valid_acc.update(acc)
                tqdm_val.set_postfix(epoch=self.epoch, 
                                    val_acc=valid_acc.avg,
                                    val_loss=valid_loss.avg)
        
        gap_val = None
        if self.calculate_GAP:
            gap_val = GAP(vals_pred, vals_conf, vals_gt)

        return valid_loss.avg, valid_acc.avg, gap_val, images_wdb


    def _reset_epochs(self):
        self.epoch = 1

    
    def update_epoch(self):
        self.epoch += 1