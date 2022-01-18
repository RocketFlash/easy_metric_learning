import os
import argparse
from os.path import isfile, join, split
from shutil import copyfile

import numpy as np
import torch
import torch.nn as nn
from torch.cuda import amp
import torchvision
import time
import json
from pathlib import Path

from tqdm.auto import tqdm
from src.dataloader import get_loader
from src.model import get_model
from src.metrics import accuracy, GAP
from src.losses import get_loss
from src.optimizers import get_optimizer
from src.schedulers import get_scheduler

from src.utils import load_ckp, save_ckp
from src.utils import load_config, Logger, get_train_val_split

from src.utils import AverageMeter, seed_everything
from src.utils import calculate_time, batch_grid

WANDB_AVAILABLE = False    
try:
    import wandb
    WANDB_AVAILABLE = True
except ModuleNotFoundError:
    print('wandb is not installed')


def train_epoch(model, train_loader, optimizer, scaler, loss_func, device='cpu', epoch=0, DEBUG=False):
    model.train()

    train_loss = AverageMeter()
    train_acc = AverageMeter()
    
    tqdm_train = tqdm(train_loader, total=int(len(train_loader)))
    images_wdb = []

    for batch_index, (data, targets) in enumerate(tqdm_train):
        batch_size = data.size(0)

        if DEBUG and batch_index>=10: break

        if not DEBUG and WANDB_AVAILABLE:
            if epoch == 0 and batch_index < 3:
                image_grid = batch_grid(data)
                save_path = os.path.join(CONFIGS["MISC"]['WORK_DIR'], f'train_batch_{batch_index}.png')
                torchvision.utils.save_image(image_grid, save_path)
                images_wdb.append(wandb.Image(save_path, caption=f'train_batch_{batch_index}'))

        data = data.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        with amp.autocast(enabled=True):
            output = model(data, targets)
            loss = loss_func(output, targets)
            acc = accuracy(output, targets) * 100
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss.update(loss.detach().item(), batch_size)
        train_acc.update(acc)

        tqdm_train.set_postfix(epoch=epoch, train_loss=train_loss.avg ,
                                            train_acc=train_acc.avg,
                                            lr=optimizer.param_groups[0]['lr'])

    return train_loss, train_acc, images_wdb


def valid_epoch(model, valid_loader, loss_func, device='cpu', epoch=0, calculate_GAP=True, DEBUG=False):
    model.eval()

    valid_loss = AverageMeter()
    valid_acc = AverageMeter()

    activation = nn.Softmax(dim=1)

    tqdm_val = tqdm(valid_loader, total=int(len(valid_loader)))
    vals_gt, vals_pred, vals_conf = [], [], []
    images_wdb = []

    with torch.no_grad():
        for batch_index, (data, targets) in enumerate(tqdm_val):
            batch_size = data.size(0)
            if DEBUG and batch_index>10: break

            if not DEBUG and WANDB_AVAILABLE:
                if epoch == 0 and batch_index < 3:
                    image_grid = batch_grid(data)
                    save_path = os.path.join(CONFIGS["MISC"]['WORK_DIR'], f'valid_batch_{batch_index}.png')
                    torchvision.utils.save_image(image_grid, save_path)
                    images_wdb.append(wandb.Image(save_path, caption=f'valid_batch_{batch_index}'))

            data = data.to(device)
            targets = targets.to(device)
            
            output = model(data, targets)

            if calculate_GAP:
                output_probs = activation(output)
                confs, pred = torch.max(output_probs, dim=1)

                vals_conf.extend(confs.cpu().numpy().tolist())
                vals_pred.extend(pred.cpu().numpy().tolist())
                vals_gt.extend(targets.cpu().numpy().tolist())

            loss = loss_func(output, targets)
            acc = accuracy(output, targets) * 100

            valid_loss.update(loss.detach().item(), batch_size)
            valid_acc.update(acc)
            tqdm_val.set_postfix(epoch=epoch, 
                                 val_acc=valid_acc.avg,
                                 val_loss=valid_loss.avg)
    
    gap_val = None
    if calculate_GAP:
        gap_val = GAP(vals_pred, vals_conf, vals_gt)
        print(f'GAP value: {gap_val}')

    return valid_loss, valid_acc, gap_val, images_wdb


def train(CONFIGS):
    wandb_run = None
    if CONFIGS['GENERAL']['USE_WANDB'] and WANDB_AVAILABLE:
        wandb_run = wandb.init(project=f'RETECHLABS metric learning',
                               name=CONFIGS["MISC"]['RUN_NAME'],
                               reinit=True)

    best_weights_name = 'debug_best.pt' if CONFIGS['GENERAL']['DEBUG'] else 'best.pt'
    last_weights_name = 'debug_last.pt' if CONFIGS['GENERAL']['DEBUG'] else 'last.pt'
    best_embeddings_weights_name = 'debug_best_emb.pt' if CONFIGS['GENERAL']['DEBUG'] else 'best_emb.pt'
    best_cpk_save_path = os.path.join(CONFIGS["MISC"]['WORK_DIR'], best_weights_name)
    last_cpk_save_path = os.path.join(CONFIGS["MISC"]['WORK_DIR'], last_weights_name)
    best_embeddings_cpk_save_path = os.path.join(CONFIGS["MISC"]['WORK_DIR'], best_embeddings_weights_name)

    start_epoch = 0

    df_train, df_valid, df_full = get_train_val_split(CONFIGS["DATA"]["SPLIT_FILE"],
                                                      fold=CONFIGS["DATA"]["FOLD"])

    train_loader = get_loader(CONFIGS["DATA"]["DIR"],
                              df_train, 
                              batch_size=CONFIGS["DATA"]["BATCH_SIZE"],
                              num_thread=CONFIGS["DATA"]["WORKERS"], 
                              img_size=CONFIGS['DATA']['IMG_SIZE'],
                              split='train',
                              transform_name=CONFIGS['TRAIN']['AUG_TYPE'])
                            
    valid_loader = get_loader(CONFIGS["DATA"]["DIR"],
                              df_valid,
                              batch_size=CONFIGS["DATA"]["BATCH_SIZE"],
                              split='val',
                              num_thread=CONFIGS["DATA"]["WORKERS"], 
                              img_size=CONFIGS['DATA']['IMG_SIZE'])

    total_n_classes = df_full['label_id'].nunique()
    train_n_classes = df_train['label_id'].nunique()
    valid_n_classes = df_valid['label_id'].nunique()

    total_n_samples = len(df_full)
    train_n_samples = len(df_train)
    valid_n_samples = len(df_valid)

    if CONFIGS['TRAIN']['AUTO_SCALE_SIZE']:
        scale_size = np.sqrt(2) * np.log(train_n_classes-1)  
    else:
        scale_size = CONFIGS['MODEL']['SCALE_SIZE']

    if CONFIGS['MODEL']['DYNAMIC_MARGIN_LAMBDA']:
        classes_counts = dict(df_full['label_id'].value_counts())
        margin_m = {}
        for class_id, class_cnt in classes_counts.items():
            margin_m[class_id] = CONFIGS['MODEL']['DYNAMIC_MARGIN_HB']*class_cnt**(-CONFIGS['MODEL']['DYNAMIC_MARGIN_LAMBDA']) + CONFIGS['MODEL']['DYNAMIC_MARGIN_LB']
    else:
        margin_m = CONFIGS['MODEL']['M']

    logger.info(f'''
    ============   TRAINING PARAMETERS   ============
    Total N classes           : {total_n_classes}
    Total N classes train     : {train_n_classes}
    Total N classes valid     : {valid_n_classes}
    Total N samples           : {total_n_samples}
    Total N training samples  : {train_n_samples}
    Total N validation samples: {valid_n_samples}
    Scale size s              : {scale_size:.2f}
    Margin m                  : {margin_m if not isinstance(margin_m, dict) else 'dynamic'}
    =================================================''')

    device = torch.device(CONFIGS['GENERAL']['DEVICE']) 

    model = get_model(model_name=CONFIGS['MODEL']['ENCODER_NAME'], 
                      margin_type=CONFIGS['MODEL']['MARGIN_TYPE'],
                      embeddings_size=CONFIGS['MODEL']['EMBEDDINGS_SIZE'],   
                      dropout=CONFIGS['TRAIN']['DROPOUT_PROB'],
                      out_features=total_n_classes,
                      scale_size=scale_size,
                      m=margin_m,
                      K=CONFIGS['MODEL']['K'],
                      easy_margin=False,
                      ls_eps=CONFIGS['TRAIN']['LS_PROB'])

    model = model.to(device)


    train_loss_func = get_loss(CONFIGS['TRAIN']['LOSS_TYPE'], 
                                gamma=CONFIGS['TRAIN']['FOCAL_GAMMA'],
                                num_classes=total_n_classes)
    train_loss_func = train_loss_func.to(device)
    test_loss_func = get_loss('cross_entropy')
    

    # optimizer
    optimizer = get_optimizer(model, CONFIGS["OPTIMIZER"])

    # learning rate scheduler
    scheduler = get_scheduler(optimizer, CONFIGS["OPTIMIZER"])


    best_loss = 100
    if CONFIGS['TRAIN']['RESUME'] is not None:

        logger.info('Resume training from: {}'.format(CONFIGS['TRAIN']['RESUME']))
        model, optimizer, start_epoch, best_loss = load_ckp(CONFIGS['TRAIN']['RESUME'], model, optimizer)
    
    if CONFIGS["TRAIN"]["LOAD_WEIGHTS"]:
        if isfile(CONFIGS["TRAIN"]["LOAD_WEIGHTS"]):
            logger.info("=> loading checkpoint '{}'".format(CONFIGS["TRAIN"]["LOAD_WEIGHTS"]))
            model, _, _, _ = load_ckp(CONFIGS["TRAIN"]["LOAD_WEIGHTS"], model)
            logger.info("=> start from checkpoint '{}'".format(CONFIGS["TRAIN"]["LOAD_WEIGHTS"]))
        else:
            logger.info("=> no checkpoint found at '{}'".format(CONFIGS["TRAIN"]["LOAD_WEIGHTS"]))
   
    print(f'Current best loss: {best_loss}')
    
    if CONFIGS['GENERAL']['USE_WANDB'] and WANDB_AVAILABLE and not CONFIGS['GENERAL']['DEBUG']:
        wandb.watch(model)

    metrics = {}
    scaler = amp.GradScaler(enabled=True)

    if model.margin.m != margin_m:
        model.margin.update(margin_m)

    start_time = time.time()
    for epoch in range(start_epoch, CONFIGS['TRAIN']['EPOCHS'] + 1):
        train_loss, train_acc, images_wdb_train = train_epoch(model, train_loader, 
                                                                optimizer, 
                                                                scaler, 
                                                                loss_func=train_loss_func,
                                                                device=device,
                                                                epoch=epoch, 
                                                                DEBUG=CONFIGS['GENERAL']['DEBUG'])
        
        save_ckp(last_cpk_save_path, model, epoch, optimizer, best_loss)
        
        metrics['train_loss'] = train_loss.avg
        metrics['train_acc'] = train_acc.avg
        metrics['learning_rate'] = optimizer.param_groups[0]['lr']
        
        epoch_info_str = f'Epoch: {epoch} Training Loss: {train_loss.avg:.5f} Training Acc: {train_acc.avg:.5f}\n'
        
        valid_loss, valid_acc, gap_val, images_wdb_valid = valid_epoch(model,valid_loader, 
                                                                    loss_func=test_loss_func, 
                                                                    epoch=epoch, 
                                                                    device=device,
                                                                    calculate_GAP=CONFIGS['TRAIN']['CALCULATE_GAP'], 
                                                                    DEBUG=CONFIGS['GENERAL']['DEBUG'])

        if images_wdb_train and images_wdb_valid:
            metrics["training batch"] = images_wdb_train
            metrics["validation batch"] = images_wdb_valid

        metrics['valid_loss'] = valid_loss.avg
        metrics['valid_acc'] = valid_acc.avg
        if gap_val is not None: metrics['gap_val'] = gap_val

        scheduler.step()

        if valid_loss.avg < best_loss:
            logger.info('Saving best model')
            best_loss = valid_loss.avg
            save_ckp(best_cpk_save_path, model, epoch, optimizer, best_loss)
            save_ckp(best_embeddings_cpk_save_path, model.embeddings_net, emb_model_only=True)

        epoch_info_str += f'         Validation Loss: {valid_loss.avg:.5f} Validation Acc: {valid_acc.avg:.5f}'
        
        if CONFIGS['GENERAL']['USE_WANDB'] and not CONFIGS['GENERAL']['DEBUG']:
            wandb.log(metrics, step=epoch)

            # print training/validation statistics
            logger.info(epoch_info_str)


        elapsed, remaining = calculate_time(start_time=start_time, 
                                            start_epoch=start_epoch, 
                                            epoch=epoch, 
                                            epochs=CONFIGS["TRAIN"]["EPOCHS"])
        
        logger.info("Epoch {0}/{1} finishied, saved to {2} .\t"
                    "Elapsed {elapsed.days:d} days {elapsed.hours:d} hours {elapsed.minutes:d} minutes.\t"
                    "Remaining {remaining.days:d} days {remaining.hours:d} hours {remaining.minutes:d} minutes.".format(
            epoch, CONFIGS["TRAIN"]["EPOCHS"], CONFIGS["MISC"]['WORK_DIR'], elapsed=elapsed, remaining=remaining))

    logger.info("Training done, all results saved to {}".format(CONFIGS["MISC"]['WORK_DIR']))

    if wandb_run is not None:
        wandb_run.finish()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='', help='path to cfg.yaml')
    parser.add_argument('--resume', default="", help='path to config file')
    parser.add_argument('--tmp', default="", help='tmp')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--device', type=str, default='', help='select device')
    args = parser.parse_args()

    assert os.path.isfile(args.config)
    CONFIGS = load_config(args.config)

    seed_everything(CONFIGS['GENERAL']['RANDOM_STATE'])

    if args.tmp != "" and args.tmp != CONFIGS["MISC"]["TMP"]:
        CONFIGS["MISC"]["TMP"] = args.tmp
    
    if args.debug: CONFIGS['GENERAL']['DEBUG'] = True
    if CONFIGS['GENERAL']['DEBUG']: 
        print('DEBUG MODE')
        CONFIGS['GENERAL']['USE_WANDB'] = False

    if args.device:
        CONFIGS['GENERAL']['DEVICE'] = int(args.device)
    

    CONFIGS["OPTIMIZER"]["WEIGHT_DECAY"] = float(CONFIGS["OPTIMIZER"]["WEIGHT_DECAY"])
    CONFIGS["OPTIMIZER"]["LR"] = float(CONFIGS["OPTIMIZER"]["LR"])

    CONFIGS["MISC"]['RUN_NAME'] = '{}_{}_fold{}_{}'.format(CONFIGS['MODEL']['MARGIN_TYPE'],
                                                        CONFIGS['MODEL']['ENCODER_NAME'],
                                                        CONFIGS["DATA"]["FOLD"],
                                                        CONFIGS['MISC']['RUN_INFO'])

    CONFIGS["MISC"]['WORK_DIR'] = os.path.join(CONFIGS["MISC"]["TMP"], 
                                               CONFIGS["MISC"]['RUN_NAME'])
    os.makedirs(CONFIGS["MISC"]['WORK_DIR'], exist_ok=True)

    logger = Logger(os.path.join(CONFIGS["MISC"]['WORK_DIR'], "log.txt"))
    logger.info(json.dumps(CONFIGS, sort_keys=False, indent=4))

    copyfile(args.config, Path(CONFIGS["MISC"]['WORK_DIR'])/'config.yml')

    train(CONFIGS)