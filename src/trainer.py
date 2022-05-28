import os
from tqdm.auto import tqdm
import torch
import torchvision
import torch.nn as nn
from torch.cuda import amp


from .utils import AverageMeter
from .utils import batch_grid
from .metrics import accuracy, GAP

try:
    import wandb
except ModuleNotFoundError:
    print('wandb is not installed')


class MLTrainer:
    def __init__(self, model, optimizer, loss_func, logger, configs, device='cpu', epoch=0, amp_scaler=None, wandb_available=True, is_debug=False):
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.logger = logger
        self.epoch = epoch
        self.wandb_available = wandb_available
        self.configs = configs
        self.device = device
        self.is_debug = is_debug
        self.amp_scaler = amp_scaler

    def train_epoch(self, train_loader):
        self.model.train()

        train_loss = AverageMeter()
        train_acc = AverageMeter()
        
        tqdm_train = tqdm(train_loader, total=int(len(train_loader)))
        images_wdb = []

        for batch_index, (data, targets) in enumerate(tqdm_train):
            batch_size = data.size(0)

            if self.is_debug and batch_index>=10: break

            if not self.is_debug and self.wandb_available:
                if self.epoch == 0 and batch_index < 3:
                    image_grid = batch_grid(data)
                    save_path = os.path.join(self.configs["MISC"]['WORK_DIR'], f'train_batch_{batch_index}.png')
                    torchvision.utils.save_image(image_grid, save_path)
                    images_wdb.append(wandb.Image(save_path, caption=f'train_batch_{batch_index}'))

            data = data.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            if self.amp_scaler is not None:
                with amp.autocast(enabled=True):
                    output = self.model(data, targets)
                    loss = self.loss_func(output, targets)
                    acc = accuracy(output, targets)
                
                self.amp_scaler.scale(loss).backward()
                self.amp_scaler.step(self.optimizer)
                self.amp_scaler.update()
            else:
                output = self.model(data, targets)
                loss = self.loss_func(output, targets)
                acc = accuracy(output, targets)

            train_loss.update(loss.detach().item(), batch_size)
            train_acc.update(acc)

            tqdm_train.set_postfix(epoch=self.epoch, train_loss=train_loss.avg ,
                                                train_acc=train_acc.avg,
                                                lr=self.optimizer.param_groups[0]['lr'])

        return train_loss.avg, train_acc.avg, images_wdb
    

    def valid_epoch(self, valid_loader, calculate_GAP=True):
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
                    if self.epoch == 0 and batch_index < 3:
                        image_grid = batch_grid(data)
                        save_path = os.path.join(self.configs["MISC"]['WORK_DIR'], f'valid_batch_{batch_index}.png')
                        torchvision.utils.save_image(image_grid, save_path)
                        images_wdb.append(wandb.Image(save_path, caption=f'valid_batch_{batch_index}'))

                data = data.to(self.device)
                targets = targets.to(self.device)
                
                output = self.model(data, targets)

                if calculate_GAP:
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
        if calculate_GAP:
            gap_val = GAP(vals_pred, vals_conf, vals_gt)

        return valid_loss.avg, valid_acc.avg, gap_val, images_wdb


    def _reset_epochs(self):
        self.epoch = 0