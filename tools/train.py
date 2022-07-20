import sys
sys.path.append("./")

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

from src.dataset import get_loader
from src.model import get_model
from src.loss import get_loss
from src.optimizers import get_optimizer
from src.schedulers import get_scheduler

from src.utils import load_ckp, save_ckp, get_cp_save_paths
from src.utils import load_config, Logger, get_train_val_split

from src.utils import  seed_everything, calculate_time, get_device
from src.utils import calculate_autoscale, calculate_dynamic_margin
from src.trainer import MLTrainer
from src.evaluator import FacesEvaluator


def train(CONFIGS, WANDB_AVAILABLE=False):
    start_epoch = 1
    best_loss = 10000
    wandb_run = None

    if CONFIGS['GENERAL']['USE_WANDB'] and WANDB_AVAILABLE:
        wandb_run = wandb.init(project=CONFIGS["MISC"]['PROJECT_NAME'],
                               name=CONFIGS["MISC"]['RUN_NAME'],
                               reinit=True)
        wandb_run.config.update(CONFIGS)

    best_cp_sp, last_cp_sp, best_emb_cp_sp = get_cp_save_paths(CONFIGS)
    VALIDATE = CONFIGS["DATA"]['SPLIT_FILE'] is not None

    if VALIDATE:
        df_train, df_valid, df_full = get_train_val_split(data_config=CONFIGS["DATA"])

        train_loader, train_dataset = get_loader(df_train, data_config=CONFIGS["DATA"], split='train')
        valid_loader, valid_dataset = get_loader(df_valid, data_config=CONFIGS["DATA"], split='val')

        n_cl_total = df_full['label_id'].nunique()
        n_cl_train = df_train['label_id'].nunique()
        n_cl_valid = df_valid['label_id'].nunique()
        n_s_total = len(df_full)
        n_s_train = len(df_train)
        n_s_valid = len(df_valid)

        classes_counts = dict(df_full['label_id'].value_counts())
    else:
        train_loader, train_dataset = get_loader(data_config=CONFIGS["DATA"], split='train',  calc_cl_count=CONFIGS['MODEL']['DYNAMIC_MARGIN'])
        n_cl_total = train_dataset.num_classes
        n_cl_train = n_cl_total
        n_cl_valid = 0 
        n_s_total = train_dataset.__len__()
        n_s_train = n_s_total
        n_s_valid = 0
        classes_counts = train_dataset.classes_counts

    
    CONFIGS['MODEL']['N_CLASSES'] = n_cl_total
    CONFIGS['TRAIN']['N_CLASSES'] = n_cl_total

    if CONFIGS['MODEL']['AUTO_SCALE_SIZE']:
        CONFIGS['MODEL']['S'] = calculate_autoscale(n_cl_total)  

    if CONFIGS['MODEL']['DYNAMIC_MARGIN']:
        CONFIGS['MODEL']['M'] = calculate_dynamic_margin(CONFIGS['MODEL']['DYNAMIC_MARGIN'], classes_counts)

    logger.data_info(CONFIGS, n_cl_total, n_cl_train, n_cl_valid, n_s_total, n_s_train, n_s_valid) 

    device = get_device(CONFIGS['GENERAL']['DEVICE'])
    model  = get_model(model_config=CONFIGS['MODEL']).to(device)

    loss_func = get_loss(train_config=CONFIGS['TRAIN']).to(device)
    optimizer = get_optimizer(model,     CONFIGS['TRAIN']["OPTIMIZER"])
    scheduler = get_scheduler(optimizer, CONFIGS['TRAIN']["SCHEDULER"])
    
    if CONFIGS['TRAIN']["WARMUP"]:
        import pytorch_warmup as warmup
        if 'adam' in CONFIGS['TRAIN']["OPTIMIZER"]:
            warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
        else:
            warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=100)
    else:
        warmup_scheduler = None

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
   
    logger.info(f'Current best loss: {best_loss}')
    
    if CONFIGS['GENERAL']['USE_WANDB'] and WANDB_AVAILABLE and not CONFIGS['GENERAL']['DEBUG']:
        wandb.watch(model)

    metrics = {}
    scaler = amp.GradScaler() if device != 'cpu' and CONFIGS["TRAIN"]['AMP'] else None

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
                        warmup_scheduler=warmup_scheduler,
                        wandb_available=WANDB_AVAILABLE, 
                        is_debug=CONFIGS['GENERAL']['DEBUG'])

    if CONFIGS['TEST']['FACES_EVALUATION']:
        faces_evaluator = FacesEvaluator(CONFIGS['TEST']['VAL_TARGETS'], 
                                         CONFIGS['DATA']['DIR'],
                                         logger=logger,
                                         device=device)
    else:
        faces_evaluator = None

    start_time = time.time()
    for epoch in range(start_epoch, CONFIGS['TRAIN']['EPOCHS'] + 1):
        train_loss, train_acc, images_wdb_train = trainer.train_epoch(train_loader)
        if VALIDATE:
            valid_loss, valid_acc, gap_val, images_wdb_valid = trainer.valid_epoch(valid_loader, calculate_GAP=CONFIGS['TRAIN']['CALCULATE_GAP'])
        else:
            valid_loss, valid_acc, gap_val = None, None, None
        trainer.update_epoch()
        
        if faces_evaluator is not None:
            faces_evaluator(model.embeddings_net)
        
        save_ckp(last_cp_sp, model, epoch, optimizer, best_loss)

        if warmup_scheduler is not None:
            with warmup_scheduler.dampening():
                scheduler.step()
        else:
            scheduler.step()
        
        check_loss = valid_loss if VALIDATE else train_loss
        
        if check_loss < best_loss:
            logger.info('Saving best model')
            best_loss = check_loss
            save_ckp(best_cp_sp, model, epoch, optimizer, best_loss)
            save_ckp(best_emb_cp_sp, model.embeddings_net, emb_model_only=True)

        if CONFIGS['GENERAL']['USE_WANDB'] and not CONFIGS['GENERAL']['DEBUG']:
            metrics['train_loss'] = train_loss
            metrics['train_acc']  = train_acc
            metrics['learning_rate'] = optimizer.param_groups[0]['lr']

            if images_wdb_train:
                metrics["training batch"] = images_wdb_train
                
            if VALIDATE:
                if images_wdb_valid:
                    metrics["validation batch"] = images_wdb_valid
                metrics['valid_loss'] = valid_loss
                metrics['valid_acc']  = valid_acc
                if gap_val is not None: metrics['gap_val'] = gap_val
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

    if args.device:
        CONFIGS['GENERAL']['DEVICE'] = int(args.device)
    
    if args.resume:
        CONFIGS['TRAIN']['RESUME'] = args.resume

    CONFIGS['TRAIN']["OPTIMIZER"]["WEIGHT_DECAY"] = float(CONFIGS['TRAIN']["OPTIMIZER"]["WEIGHT_DECAY"])
    CONFIGS['TRAIN']["OPTIMIZER"]["LR"] = float(CONFIGS['TRAIN']["OPTIMIZER"]["LR"])

    CONFIGS["MISC"]['RUN_NAME'] = '{}_{}_{}_{}'.format(CONFIGS['MISC']['DATASET_INFO'],
                                                       CONFIGS['MODEL']['MARGIN_TYPE'],
                                                       CONFIGS['MODEL']['ENCODER_NAME'],
                                                       CONFIGS['MISC']['RUN_INFO'])
    if CONFIGS["DATA"]['SPLIT_FILE'] is not None:
        CONFIGS["MISC"]['RUN_NAME'] += '_fold{}'.format(CONFIGS["DATA"]['FOLD'])

    CONFIGS["MISC"]['WORK_DIR'] = os.path.join(CONFIGS["MISC"]["TMP"], 
                                               CONFIGS["MISC"]['RUN_NAME'])
    os.makedirs(CONFIGS["MISC"]['WORK_DIR'], exist_ok=True)

    logger = Logger(os.path.join(CONFIGS["MISC"]['WORK_DIR'], "log.txt"))
    logger.info(json.dumps(CONFIGS, sort_keys=False, indent=4))

    copyfile(args.config, Path(CONFIGS["MISC"]['WORK_DIR'])/'config.yml')

    if args.debug: CONFIGS['GENERAL']['DEBUG'] = True
    if CONFIGS['GENERAL']['DEBUG']: 
        logger.info('DEBUG MODE')
        CONFIGS['GENERAL']['USE_WANDB'] = False

    WANDB_AVAILABLE = False    
    try:
        import wandb
        WANDB_AVAILABLE = True
    except ModuleNotFoundError:
        logger.info('wandb is not installed')

    train(CONFIGS, WANDB_AVAILABLE)