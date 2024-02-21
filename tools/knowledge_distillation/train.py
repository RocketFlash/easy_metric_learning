import sys
sys.path.append("./")

import os
import argparse
from os.path import isfile
from shutil import copyfile

import numpy as np
import pandas as pd
import torch
from torch.cuda import amp
import time
import json
from pathlib import Path

from src.dataset import get_loader
from src.optimizer import get_optimizer
from src.scheduler.schedulers import get_scheduler

from src.utils import load_ckp, save_ckp, get_cp_save_paths
from src.utils import load_config, Logger, get_train_val_split

from src.utils import seed_everything, get_device
from src.utils import get_value_if_exist
from src.trainer import DistillationTrainer
from src.model import get_model_embeddings


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='', help='path to cfg.yaml')
    parser.add_argument('--resume', default="", help='path to weights from which to resume')
    parser.add_argument('--tmp', default="", help='tmp')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--device', type=str, default='', help='select device')
    parser.add_argument('--teacher_config', 
                        type=str, 
                        default='', 
                        help='Knowledge distillation, path to teacher network config')
    parser.add_argument('--teacher_weights', 
                        type=str, 
                        default='', 
                        help='Knowledge distillation, path to teacher network weights')
    return parser.parse_args()


def train(CONFIGS, 
          WANDB_AVAILABLE=False,
          model_teacher=None):
    start_epoch = 1
    best_loss = 10000
    wandb_run = None

    if CONFIGS['GENERAL']['USE_WANDB'] and WANDB_AVAILABLE:
        wandb_run = wandb.init(project=CONFIGS["MISC"]['PROJECT_NAME'],
                               name=CONFIGS["MISC"]['RUN_NAME'],
                               reinit=True)
        wandb_run.config.update(CONFIGS)

    _, _, best_emb_cp_sp, last_emb_cp_sp = get_cp_save_paths(CONFIGS)
    
    labels_to_ids = None
    VALIDATE = CONFIGS["DATA"]['SPLIT_FILE'] is not None
    if VALIDATE:
        df_train, df_valid, df_full = get_train_val_split(data_config=CONFIGS["DATA"])

        train_loader, train_dataset = get_loader(df_train, 
                                                 data_config=CONFIGS["DATA"], 
                                                 split='train')
        labels_to_ids = train_dataset.get_labels_to_ids()
        
        with open(Path(CONFIGS["MISC"]['WORK_DIR']) / f'labels_to_ids.json', 'w') as fp:
            json.dump(labels_to_ids, fp)

        valid_loader, valid_dataset = get_loader(df_valid, 
                                                 data_config=CONFIGS["DATA"], 
                                                 split='val',
                                                 labels_to_ids=labels_to_ids)

        if isinstance(df_train, list):
            df_train = pd.concat(df_train, ignore_index=True, sort=False)
        if isinstance(df_valid, list):
            df_valid = pd.concat(df_valid, ignore_index=True, sort=False)
        
        n_cl_total = df_full['label'].nunique()
        n_cl_train = df_train['label'].nunique()
        n_cl_valid = df_valid['label'].nunique()
        n_s_total = len(df_full)
        n_s_train = len(df_train)
        n_s_valid = len(df_valid)
    else:
        train_loader, train_dataset = get_loader(data_config=CONFIGS["DATA"], 
                                                 split='train',  
                                                 calc_cl_count=CONFIGS['MODEL']['DYNAMIC_MARGIN'])
        n_cl_total = train_dataset.num_classes
        n_cl_train = n_cl_total
        n_cl_valid = 0 
        n_s_total = train_dataset.__len__()
        n_s_train = n_s_total
        n_s_valid = 0
    
    if n_s_valid==0:
        VALIDATE = False
        
    CONFIGS['MODEL']['N_CLASSES'] = n_cl_total
    CONFIGS['TRAIN']['N_CLASSES'] = n_cl_total

    logger.data_info(CONFIGS, 
                     n_cl_total, 
                     n_cl_train, 
                     n_cl_valid, 
                     n_s_total, 
                     n_s_train, 
                     n_s_valid) 

    device = get_device(CONFIGS['GENERAL']['DEVICE'])
    model  = get_model_embeddings(model_config=CONFIGS['MODEL']).to(device)

    if model_teacher is not None:
        model_teacher.to(device)
        model_teacher.eval()

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
        model, optimizer, epoch_resume, best_loss = load_ckp(CONFIGS['TRAIN']['RESUME'], model, optimizer)
        start_epoch = epoch_resume + 1

    if CONFIGS["TRAIN"]["LOAD_EMBEDDER"]:
        if isfile(CONFIGS["TRAIN"]["LOAD_EMBEDDER"]):
            logger.info("loading checkpoint :'{}'".format(CONFIGS["TRAIN"]["LOAD_EMBEDDER"]))
            model.embeddings_net = load_ckp(CONFIGS["TRAIN"]["LOAD_EMBEDDER"], model.embeddings_net,  emb_model_only=True)
            logger.info("start from checkpoint :'{}'".format(CONFIGS["TRAIN"]["LOAD_EMBEDDER"]))
        else:
            logger.info("no checkpoint found at :'{}'".format(CONFIGS["TRAIN"]["LOAD_EMBEDDER"]))
   
    logger.info(f'Current best loss: {best_loss}')
    
    if CONFIGS['GENERAL']['USE_WANDB'] and WANDB_AVAILABLE and not CONFIGS['GENERAL']['DEBUG']:
        wandb.watch(model)

    metrics = {}
    scaler = amp.GradScaler() if device != 'cpu' and CONFIGS["TRAIN"]['AMP'] else None

    n_train_epochs = CONFIGS['TRAIN']['EPOCHS'] - start_epoch + 1
    p_mixup   = get_value_if_exist(CONFIGS['DATA'], 'P_MIXUP',  0) 
    p_cutmix  = get_value_if_exist(CONFIGS['DATA'], 'P_CUTMIX', 0) 
    vis_batch = get_value_if_exist(CONFIGS['DATA'], 'VISUALIZE_BATCH') 

    trainer = DistillationTrainer(model=model, 
                                  model_teacher=model_teacher,
                                  optimizer=optimizer, 
                                  logger=logger, 
                                  device=device,
                                  epoch=start_epoch,
                                  amp_scaler=scaler,
                                  warmup_scheduler=warmup_scheduler,
                                  wandb_available=WANDB_AVAILABLE, 
                                  is_debug=CONFIGS['GENERAL']['DEBUG'],
                                  grad_accum_steps=CONFIGS['TRAIN']['GRADIENT_ACC_STEPS'],
                                  work_dir=CONFIGS["MISC"]['WORK_DIR'],
                                  visualize_batch=vis_batch,
                                  n_epochs=n_train_epochs,
                                  p_mixup=p_mixup,
                                  p_cutmix=p_cutmix)

    start_time = time.time()
    for epoch in range(start_epoch, CONFIGS['TRAIN']['EPOCHS'] + 1):
        stats_train = trainer.train_epoch(train_loader)

        stats_valid = None
        if VALIDATE:
            stats_valid = trainer.valid_epoch(valid_loader)
            
        trainer.update_epoch()
        save_ckp(last_emb_cp_sp, model, emb_model_only=True)

        if warmup_scheduler is not None:
            with warmup_scheduler.dampening():
                scheduler.step()
        else:
            scheduler.step()
        
        check_loss = stats_valid.loss if VALIDATE else stats_train.loss
        
        if check_loss < best_loss:
            logger.info('Saving best model')
            best_loss = check_loss
            save_ckp(best_emb_cp_sp, model, emb_model_only=True)

        if CONFIGS['GENERAL']['USE_WANDB'] and not CONFIGS['GENERAL']['DEBUG']:
            metrics['train_loss'] = stats_train.loss
            metrics['learning_rate'] = optimizer.param_groups[-1]['lr']

            if stats_train.images_wdb:
                metrics["training batch"] = stats_train.images_wdb
                
            if VALIDATE:
                if stats_valid.images_wdb:
                    metrics["validation batch"] = stats_valid.images_wdb
                metrics['valid_loss'] = stats_valid.loss
            wandb.log(metrics, step=epoch)
        
        logger.epoch_time_info(start_time, 
                               start_epoch, 
                               epoch, 
                               num_epochs=CONFIGS["TRAIN"]["EPOCHS"], 
                               workdir_path=CONFIGS["MISC"]['WORK_DIR'])

    logger.info("Training done, all results saved to {}".format(CONFIGS["MISC"]['WORK_DIR']))

    if wandb_run is not None:
        wandb_run.finish()


if __name__=="__main__":
    args = parse_args()
    assert os.path.isfile(args.config)
    CONFIGS = load_config(args.config)

    seed_everything(CONFIGS['GENERAL']['RANDOM_STATE'])

    if args.tmp != "" and args.tmp != CONFIGS["MISC"]["TMP"]:
        CONFIGS["MISC"]["TMP"] = args.tmp

    if args.device:
        CONFIGS['GENERAL']['DEVICE'] = int(args.device)
    
    if args.resume:
        CONFIGS['TRAIN']['RESUME'] = args.resume

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

    model_teacher = None
    if args.teacher_config and args.teacher_weights:
        assert os.path.isfile(args.teacher_config)
        CONFIGS_T = load_config(args.teacher_config)
        model_teacher = get_model_embeddings(model_config=CONFIGS_T['MODEL'])
        model_teacher = load_ckp(args.teacher_weights, 
                                 model_teacher, 
                                 emb_model_only=True)
        for param in model_teacher.parameters():
            param.requires_grad = False

    train(CONFIGS, 
          WANDB_AVAILABLE,
          model_teacher)