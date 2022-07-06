import sys
sys.path.append("..")

import os
import argparse
from os.path import isfile
from shutil import copyfile

import numpy as np
import torch
from torch.cuda import amp
import time
import json
from pathlib import Path

from src.dataloader import get_loader
from src.model import get_model
from src.loss import get_loss
from src.optimizers import get_optimizer
from src.schedulers import get_scheduler

from src.utils import load_ckp, save_ckp, get_cp_save_paths
from src.utils import load_config, Logger, get_train_val_split

from src.utils import  seed_everything, calculate_time
from src.utils import calculate_autoscale
from src.trainer import MLTrainer

WANDB_AVAILABLE = False    
try:
    import wandb
    WANDB_AVAILABLE = True
except ModuleNotFoundError:
    print('wandb is not installed')



def train(CONFIGS):
    start_epoch = 1
    wandb_run = None

    if CONFIGS['GENERAL']['USE_WANDB'] and WANDB_AVAILABLE:
        wandb_run = wandb.init(project=f'RETECHLABS metric learning',
                               name=CONFIGS["MISC"]['RUN_NAME'],
                               reinit=True)
        wandb_run.config.update(CONFIGS)

    best_cp_sp, last_cp_sp, best_emb_cp_sp = get_cp_save_paths(CONFIGS)

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

    if CONFIGS['TRAIN']['AUTO_SCALE_SIZE']:
        CONFIGS['MODEL']['SCALE_SIZE'] = calculate_autoscale(train_n_classes)  

    if CONFIGS['MODEL']['DYNAMIC_MARGIN_LAMBDA']:
        classes_counts = dict(df_full['label_id'].value_counts())
        CONFIGS['MODEL']['M'] = {}
        for class_id, class_cnt in classes_counts.items():
            CONFIGS['MODEL']['M'][class_id] = CONFIGS['MODEL']['DYNAMIC_MARGIN_HB']*class_cnt**(-CONFIGS['MODEL']['DYNAMIC_MARGIN_LAMBDA']) + CONFIGS['MODEL']['DYNAMIC_MARGIN_LB']

    device = torch.device(CONFIGS['GENERAL']['DEVICE'])

    logger.data_info(CONFIGS, df_full, df_train, df_valid) 

    model = get_model(model_name=CONFIGS['MODEL']['ENCODER_NAME'], 
                      margin_type=CONFIGS['MODEL']['MARGIN_TYPE'],
                      embeddings_size=CONFIGS['MODEL']['EMBEDDINGS_SIZE'],   
                      dropout=CONFIGS['TRAIN']['DROPOUT_PROB'],
                      out_features=total_n_classes,
                      scale_size=CONFIGS['MODEL']['SCALE_SIZE'],
                      m=CONFIGS['MODEL']['M'],
                      K=CONFIGS['MODEL']['K'],
                      easy_margin=False,
                      ls_eps=CONFIGS['TRAIN']['LS_PROB']).to(device)

    loss_func = get_loss(CONFIGS['TRAIN']['LOSS_TYPE'], 
                         gamma=CONFIGS['TRAIN']['FOCAL_GAMMA'],
                         num_classes=total_n_classes).to(device)
    
    
    optimizer = get_optimizer(model, CONFIGS["OPTIMIZER"])
    scheduler = get_scheduler(optimizer, CONFIGS["OPTIMIZER"])

    best_loss = 10000
    if CONFIGS['TRAIN']['RESUME'] is not None:
        logger.info('resume training from: {}'.format(CONFIGS['TRAIN']['RESUME']))
        model, optimizer, start_epoch, best_loss = load_ckp(CONFIGS['TRAIN']['RESUME'], model, optimizer)
    
    if CONFIGS["TRAIN"]["LOAD_WEIGHTS"]:
        if isfile(CONFIGS["TRAIN"]["LOAD_WEIGHTS"]):
            logger.info("loading checkpoint :'{}'".format(CONFIGS["TRAIN"]["LOAD_WEIGHTS"]))
            model, _, _, _ = load_ckp(CONFIGS["TRAIN"]["LOAD_WEIGHTS"], model)
            logger.info("start from checkpoint :'{}'".format(CONFIGS["TRAIN"]["LOAD_WEIGHTS"]))
        else:
            logger.info("no checkpoint found at :'{}'".format(CONFIGS["TRAIN"]["LOAD_WEIGHTS"]))
   
    print(f'Current best loss: {best_loss}')
    
    if CONFIGS['GENERAL']['USE_WANDB'] and WANDB_AVAILABLE and not CONFIGS['GENERAL']['DEBUG']:
        wandb.watch(model)

    metrics = {}
    scaler = amp.GradScaler(enabled=True) if device != 'cpu' else None

    if model.margin.m != CONFIGS['MODEL']['M']:
        model.margin.update(CONFIGS['MODEL']['M'])

    trainer = MLTrainer(model=model, 
                        optimizer=optimizer, 
                        loss_func=loss_func, 
                        logger=logger, 
                        configs=CONFIGS, 
                        device=device, 
                        epoch=start_epoch, 
                        amp_scaler=scaler,
                        wandb_available=WANDB_AVAILABLE, 
                        is_debug=CONFIGS['GENERAL']['DEBUG'])

    start_time = time.time()
    for epoch in range(start_epoch, CONFIGS['TRAIN']['EPOCHS'] + 1):
        train_loss, train_acc, images_wdb_train = trainer.train_epoch(train_loader)
        valid_loss, valid_acc, gap_val, images_wdb_valid = trainer.valid_epoch(valid_loader, calculate_GAP=CONFIGS['TRAIN']['CALCULATE_GAP'])
        
        save_ckp(last_cp_sp, model, epoch, optimizer, best_loss)
        
        metrics['train_loss'] = train_loss
        metrics['train_acc'] = train_acc
        metrics['learning_rate'] = optimizer.param_groups[0]['lr']

        if images_wdb_train and images_wdb_valid:
            metrics["training batch"] = images_wdb_train
            metrics["validation batch"] = images_wdb_valid

        metrics['valid_loss'] = valid_loss
        metrics['valid_acc'] = valid_acc
        if gap_val is not None: metrics['gap_val'] = gap_val

        scheduler.step()

        if valid_loss < best_loss:
            logger.info('Saving best model')
            best_loss = valid_loss
            save_ckp(best_cp_sp, model, epoch, optimizer, best_loss)
            save_ckp(best_emb_cp_sp, model.embeddings_net, emb_model_only=True)

        if CONFIGS['GENERAL']['USE_WANDB'] and not CONFIGS['GENERAL']['DEBUG']:
            wandb.log(metrics, step=epoch)
        
        logger.epoch_train_info(epoch, train_loss, train_acc, valid_loss, valid_acc, gap_val)
        logger.epoch_time_info(start_time, start_epoch, epoch, num_epochs=CONFIGS["TRAIN"]["EPOCHS"], 
                                                               workdir_path=CONFIGS["MISC"]['WORK_DIR'])

    logger.info("Training done, all results saved to {}".format(CONFIGS["MISC"]['WORK_DIR']))

    if wandb_run is not None:
        wandb_run.finish()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='', help='path to cfg.yaml')
    parser.add_argument('--resume', default="", help='path to weights from which to resume')
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
    
    if args.resume:
        CONFIGS['TRAIN']['RESUME'] = args.resume

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