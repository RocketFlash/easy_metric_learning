import os
from tqdm.auto import tqdm
import torch
import torchvision
import torch.nn as nn
from torch.cuda import amp
from easydict import EasyDict as edict

from .utils import AverageMeter
from .utils import batch_grid
from .metric import accuracy, GAP, f_score
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
                 incremental_margin=None,
                 visualize_batch=False):
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
        self.visualize_batch = visualize_batch
        self.loss_cat = torch.nn.BCEWithLogitsLoss()
        self.cat_loss_weight = 100
        
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

        meter_loss_total  = AverageMeter()
        meter_loss_margin = AverageMeter()
        meter_loss_cat    = AverageMeter()
        meter_acc         = AverageMeter()
        meter_acc_cat     = AverageMeter()
        
        tqdm_train = tqdm(train_loader, 
                          total=int(len(train_loader)))
        images_wdb = []
        with_categories = False

        for batch_index, (data, targets) in enumerate(tqdm_train):
            if self.is_debug and batch_index>=10: break

            if self.visualize_batch:
                if not self.is_debug and self.wandb_available:
                    if self.epoch == 1 and batch_index == 0:
                        image_grid = batch_grid(data)
                        save_path = os.path.join(self.work_dir, 
                                                f'train_batch_{batch_index}.png')
                        torchvision.utils.save_image(image_grid, save_path)
                        images_wdb.append(wandb.Image(save_path, 
                                                    caption=f'train_batch_{batch_index}'))

            data = data.to(self.device)
            if isinstance(targets, list):
                with_categories = True
                targets = [t.to(self.device) for t in targets]
            else:
                targets = targets.to(self.device)

            loss_c = None
            if self.amp_scaler is not None:
                with amp.autocast():
                    output = self.model(data, targets)
                    if isinstance(output, list):
                        loss_m = self.loss_func(output[0], targets[0])
                        loss_c = self.loss_cat(output[1], targets[1])
                        loss = loss_m + (loss_c * self.cat_loss_weight)
                        acc = accuracy(output[0], targets[0])
                        acc_cat = f_score(output[1], targets[1])
                        loss_m /= self.grad_accum_steps
                        loss_c /= self.grad_accum_steps
                    else:
                        loss_m = self.loss_func(output, targets)
                        loss = loss_m
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
                if isinstance(output, list):
                    loss_m = self.loss_func(output[0], targets[0])
                    loss_c = self.loss_cat(output[1], targets[1])
                    loss = loss_m + (loss_c * self.cat_loss_weight)
                    acc = accuracy(output[0], targets[0])
                    acc_cat = f_score(output[1], targets[1])
                    loss_m /= self.grad_accum_steps
                    loss_c /= self.grad_accum_steps
                else:
                    loss_m = self.loss_func(output, targets)
                    loss = loss_m
                    acc = accuracy(output, targets)
                loss /= self.grad_accum_steps
                loss.backward()

                if ((batch_index + 1) % self.grad_accum_steps == 0) or (batch_index + 1 == len(train_loader)):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            meter_loss_total.update(loss.detach().item())
            meter_acc.update(acc)
            if loss_c is not None:
                meter_acc_cat.update(acc_cat.detach().item())
                meter_loss_margin.update(loss_m.detach().item())
                meter_loss_cat.update(loss_c.detach().item())

            if self.warmup_scheduler is not None:
                if batch_index < len(tqdm_train)-1:
                    with self.warmup_scheduler.dampening(): pass

            info_params = dict(
                epoch=self.epoch, 
                loss=meter_loss_total.avg ,
                acc=meter_acc.avg,
                lr=self.optimizer.param_groups[-1]['lr'],
                m=self.model.margin.m
            )
            if loss_c is not None:
                info_params['f1_cat'] = meter_acc_cat.avg
                info_params['loss_margin'] = meter_loss_margin.avg
                info_params['loss_cat'] = meter_loss_cat.avg

            tqdm_train.set_postfix(**info_params)

        stats = dict(
            loss=meter_loss_total.avg,
            acc=meter_acc.avg,
            images_wdb=images_wdb,
            f1_cat=None,
            loss_cat=None,
            loss_margin=None
        )

        if with_categories:
            stats['f1_cat'] = meter_acc_cat.avg
            stats['loss_cat'] = meter_loss_cat.avg
            stats['loss_margin'] = meter_loss_margin.avg

        return edict(stats)
    

    def valid_epoch(self, valid_loader):
        self.model.eval()

        meter_loss_total  = AverageMeter()
        meter_loss_margin = AverageMeter()
        meter_loss_cat    = AverageMeter()
        meter_acc         = AverageMeter()
        meter_acc_cat     = AverageMeter()

        activation = nn.Softmax(dim=1)

        tqdm_val = tqdm(valid_loader, total=int(len(valid_loader)))
        vals_gt, vals_pred, vals_conf = [], [], []
        images_wdb = []
        with_categories = False

        with torch.no_grad():
            for batch_index, (data, targets) in enumerate(tqdm_val):
                batch_size = data.size(0)
                if self.is_debug and batch_index>10: break

                if self.visualize_batch:
                    if not self.is_debug and self.wandb_available:
                        if self.epoch == 1 and batch_index == 0:
                            image_grid = batch_grid(data)
                            save_path = os.path.join(self.work_dir, 
                                                    f'valid_batch_{batch_index}.png')
                            torchvision.utils.save_image(image_grid, save_path)
                            images_wdb.append(wandb.Image(save_path, 
                                                        caption=f'valid_batch_{batch_index}'))

                loss_c = None
                data = data.to(self.device)
                if isinstance(targets, list):
                    with_categories = True
                    targets = [t.to(self.device) for t in targets]
                else:
                    targets = targets.to(self.device)
                
                output = self.model(data, targets)
                if isinstance(output, list):
                    loss_m = self.loss_func(output[0], targets[0])
                    loss_c = self.loss_cat(output[1], targets[1])
                    loss = loss_m + (loss_c * self.cat_loss_weight)
                    acc = accuracy(output[0], targets[0])
                    acc_cat = f_score(output[1], targets[1])
                    g_out = output[0]
                else:
                    loss_m = self.loss_func(output, targets)
                    loss = loss_m
                    acc = accuracy(output, targets)
                    g_out = output

                if self.calculate_GAP:
                    output_probs = activation(g_out)
                    confs, pred = torch.max(output_probs, dim=1)

                    vals_conf.extend(confs.cpu().numpy().tolist())
                    vals_pred.extend(pred.cpu().numpy().tolist())
                    vals_gt.extend(targets.cpu().numpy().tolist())


                meter_loss_total.update(loss.detach().item())
                meter_acc.update(acc)
                if loss_c is not None:
                    meter_acc_cat.update(acc_cat.detach().item())
                    meter_loss_margin.update(loss_m.detach().item())
                    meter_loss_cat.update(loss_c.detach().item())

                info_params = dict(
                    epoch=self.epoch, 
                    loss=meter_loss_total.avg ,
                    acc=meter_acc.avg,
                )
                if loss_c is not None:
                    info_params['f1_cat'] = meter_acc_cat.avg
                    info_params['loss_margin'] = meter_loss_margin.avg
                    info_params['loss_cat'] = meter_loss_cat.avg

                tqdm_val.set_postfix(**info_params)
        
        stats = dict(
            loss=meter_loss_total.avg,
            acc=meter_acc.avg,
            images_wdb=images_wdb,
            f1_cat=None,
            loss_cat=None,
            loss_margin=None
        )

        gap_val = None
        if self.calculate_GAP:
            gap_val = GAP(vals_pred, vals_conf, vals_gt)
        stats['gap'] = gap_val

        if with_categories:
            stats['f1_cat'] = meter_acc_cat.avg
            stats['loss_cat'] = meter_loss_cat.avg
            stats['loss_margin'] = meter_loss_margin.avg

        return edict(stats)


    def _reset_epochs(self):
        self.epoch = 1

    
    def update_epoch(self):
        self.epoch += 1