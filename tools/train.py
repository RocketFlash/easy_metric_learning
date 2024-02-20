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

import hydra
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig

from src.dataset import get_loader
from src.model import get_model
from src.loss import get_loss
from src.optimizers import get_optimizer
from src.schedulers import get_scheduler

from src.data.utils import get_train_val_split

from src.utils import load_ckp, save_ckp, get_cp_save_paths
from src.utils import load_config, Logger

from src.utils import seed_everything, get_device
from src.utils import calculate_autoscale, calculate_dynamic_margin
from src.utils import get_value_if_exist
from src.trainer import MLTrainer
from src.experiment_tracker import (WandbTracker, 
                                    MLFlowTracker)


@hydra.main(version_base=None,
            config_path='../configs/',
            config_name='config')
def train(config):
    seed_everything(config.random_state)

    work_dir = Path(config.work_dirs) / config.run_name
    work_dir.mkdir(exist_ok=True)

    logger = Logger(work_dir / "log.txt")
    OmegaConf.save(config, work_dir / "config.yaml")
    config_dict = OmegaConf.to_container(
        config, 
        resolve=True
    )
    logger.info(
        json.dumps(
            config_dict, 
            sort_keys=False, 
            indent=4
        )
    )

    if config.debug:
        logger.info('DEBUG MODE')

    start_epoch = 1
    exp_trackers = {}
    best_loss = 10000
    labels_to_ids = None
    
    if not config.debug:
        if config.use_wandb:
            exp_trackers['wandb']  = WandbTracker(config,
                                                  config_dict)
        
        if config.use_mlflow:
            exp_trackers['mlflow'] = MLFlowTracker(config,
                                                   config_dict)
    
    (best_cp_sp, 
     last_cp_sp, 
     best_emb_cp_sp, 
     last_emb_cp_sp) = get_cp_save_paths(config, 
                                         work_dir)
        
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
    classes_counts = dict(df_full['label'].value_counts())

    device = get_device(config.device)
    model = get_model(
        config.backbone,
        config.head,
        margin_config=config.margin,
        n_classes=n_cl_total
    ).to(device)

    if CONFIGS['MODEL']['AUTO_SCALE_SIZE']:
        CONFIGS['MODEL']['S'] = calculate_autoscale(n_cl_total)  

    if CONFIGS['MODEL']['DYNAMIC_MARGIN'] is not None:
        CONFIGS['MODEL']['M'] = calculate_dynamic_margin(CONFIGS['MODEL']['DYNAMIC_MARGIN'], 
                                                         classes_counts,
                                                         labels_to_ids=labels_to_ids)

    logger.data_info(CONFIGS, 
                     n_cl_total, 
                     n_cl_train, 
                     n_cl_valid, 
                     n_s_total, 
                     n_s_train, 
                     n_s_valid) 

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
        model, optimizer, epoch_resume, best_loss = load_ckp(CONFIGS['TRAIN']['RESUME'], model, optimizer)
        start_epoch = epoch_resume + 1
    
    if CONFIGS["TRAIN"]["LOAD_WEIGHTS"]:
        if isfile(CONFIGS["TRAIN"]["LOAD_WEIGHTS"]):
            logger.info("loading checkpoint :'{}'".format(CONFIGS["TRAIN"]["LOAD_WEIGHTS"]))
            model, _, _, _ = load_ckp(CONFIGS["TRAIN"]["LOAD_WEIGHTS"], model)
            logger.info("start from checkpoint :'{}'".format(CONFIGS["TRAIN"]["LOAD_WEIGHTS"]))
        else:
            logger.info("no checkpoint found at :'{}'".format(CONFIGS["TRAIN"]["LOAD_WEIGHTS"]))

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

    if model.margin.m != CONFIGS['MODEL']['M']:
        model.margin.update(CONFIGS['MODEL']['M'])

    n_train_epochs = CONFIGS['TRAIN']['EPOCHS'] - start_epoch + 1
    p_mixup   = get_value_if_exist(CONFIGS['DATA'], 'P_MIXUP',  0) 
    p_cutmix  = get_value_if_exist(CONFIGS['DATA'], 'P_CUTMIX', 0) 
    vis_batch = get_value_if_exist(CONFIGS['DATA'], 'VISUALIZE_BATCH') 
    distill_loss_weight = get_value_if_exist(CONFIGS['TRAIN'], 'DISTILL_LOSS_W', 1)

    trainer = MLTrainer(model=model, 
                        optimizer=optimizer, 
                        loss_func=loss_func, 
                        logger=logger, 
                        device=device,
                        epoch=start_epoch,
                        amp_scaler=scaler,
                        warmup_scheduler=warmup_scheduler,
                        wandb_available=WANDB_AVAILABLE, 
                        is_debug=CONFIGS['GENERAL']['DEBUG'],
                        calculate_GAP=CONFIGS['TRAIN']['CALCULATE_GAP'],
                        grad_accum_steps=CONFIGS['TRAIN']['GRADIENT_ACC_STEPS'],
                        incremental_margin=CONFIGS['TRAIN']['INCREMENTAL_MARGIN'],
                        work_dir=CONFIGS["MISC"]['WORK_DIR'],
                        visualize_batch=vis_batch,
                        n_epochs=n_train_epochs,
                        p_mixup=p_mixup,
                        p_cutmix=p_cutmix,
                        model_teacher=model_teacher,
                        distill_loss_weight=distill_loss_weight)

    start_time = time.time()
    for epoch in range(start_epoch, CONFIGS['TRAIN']['EPOCHS'] + 1):
        stats_train = trainer.train_epoch(train_loader)

        stats_valid = None            
        trainer.update_epoch()
        
        save_ckp(last_cp_sp, model, epoch, optimizer, best_loss)
        save_ckp(last_emb_cp_sp, model.embeddings_net, emb_model_only=True)

        if warmup_scheduler is not None:
            with warmup_scheduler.dampening():
                scheduler.step()
        else:
            scheduler.step()
        
        check_loss = stats_valid.loss
        
        if check_loss < best_loss:
            logger.info('Saving best model')
            best_loss = check_loss
            save_ckp(best_cp_sp, model, epoch, optimizer, best_loss)
            save_ckp(best_emb_cp_sp, model.embeddings_net, emb_model_only=True)

        if CONFIGS['GENERAL']['USE_WANDB'] and not CONFIGS['GENERAL']['DEBUG']:
            metrics['train_loss'] = stats_train.loss
            metrics['train_acc']  = stats_train.acc
            metrics['learning_rate'] = optimizer.param_groups[-1]['lr']
            
            if stats_train.images_wdb:
                metrics["training batch"] = stats_train.images_wdb
           
            if stats_valid.images_wdb:
                metrics["validation batch"] = stats_valid.images_wdb
            metrics['valid_loss'] = stats_valid.loss
            metrics['valid_acc']  = stats_valid.acc
            if stats_valid.gap is not None: metrics['gap'] = stats_valid.gap
                
            wandb.log(metrics, step=epoch)
        
        logger.epoch_train_info(epoch, 
                                stats_train,
                                stats_valid)
        logger.epoch_time_info(start_time, 
                               start_epoch, 
                               epoch, 
                               num_epochs=CONFIGS["TRAIN"]["EPOCHS"], 
                               workdir_path=CONFIGS["MISC"]['WORK_DIR'])

    logger.info("Training done, all results saved to {}".format(CONFIGS["MISC"]['WORK_DIR']))

    if wandb_run is not None:
        wandb_run.finish()


if __name__=="__main__":
    train()