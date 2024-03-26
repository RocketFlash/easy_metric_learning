import sys
sys.path.append("./")

import time
import shutil
import hydra
import multiprocessing
from pathlib import Path
from omegaconf import OmegaConf

from src.data import (get_train_data_from_config,
                      get_test_data_from_config)
from src.model import get_model
from src.optimizer import get_optimizer
from src.logger import Logger
from src.data.utils import save_labels_to_ids
from src.trainer import get_trainer
from src.evaluator import get_evaluator
from src.experiment_tracker import get_experiment_trackers
from src.utils import (load_checkpoint, 
                       save_ckp, 
                       get_save_paths, 
                       seed_everything, 
                       get_device,
                       is_model_best,
                       is_main_process)


@hydra.main(version_base=None,
            config_path='../configs/',
            config_name='config_train')
def train(config):
    seed_everything(config.random_state)

    if config.n_workers=='auto':
        config.n_workers = multiprocessing.cpu_count()

    accelerator = None
    if config.ddp:
        from accelerate import Accelerator
        from accelerate.utils import set_seed
        set_seed(config.random_state)
        mixed_precision = "fp16" if config.amp else 'no' 
        accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=config.train.trainer.grad_accum_steps,
            step_scheduler_with_optimizer=False
        )
        config.optimizer.optimizer.lr *= accelerator.num_processes

    work_dir = Path(config.work_dirs) / config.run_name
    work_dir.mkdir(exist_ok=True)

    logger = Logger(
        work_dir / "log.txt",
        accelerator=accelerator
    )
    logger.info_config(config)
    if config.debug: logger.info('DEBUG MODE')
    OmegaConf.save(config, work_dir / "config_train.yaml")
    
    best_criterion_val = 10000 if config.train.best_model_criterion.type == 'loss' else -10000
    start_epoch = 1
    
    save_paths = get_save_paths(work_dir)
    data_info_train = get_train_data_from_config(config, logger=logger)
    data_infos_test = get_test_data_from_config(config, logger=logger)

    train_loader = data_info_train.train.dataloader
    valid_loader = data_info_train.valid.dataloader

    logger.info_data(data_info_train.train.dataset_stats) 
    logger.info_data(data_info_train.valid.dataset_stats) 

    save_labels_to_ids(data_info_train.train.labels_to_ids, save_dir=work_dir)
    config.margin.id_counts = data_info_train.train.dataset_stats.id_counts
    
    device = get_device(config.device)
    model = get_model(
        config_backbone=config.backbone,
        config_head=config.head,
        config_margin=config.margin,
        n_classes=data_info_train.train.dataset_stats.n_classes
    )

    if config.head.type=='no_head':
        config.embeddings_size = model.embeddings_net.backbone_out_feats
        logger.info(f'Embeddings size changed to backbone output size {config.embeddings_size}')

    logger.info_model(config)

    optimizer = get_optimizer(
        model=model, 
        optimizer_config=config.optimizer
    )
    
    if config.load_checkpoint is not None:
        checkpoint_data = load_checkpoint(
            config.load_checkpoint, 
            model=model, 
            optimizer=optimizer, 
            logger=logger, 
            mode=config.load_mode,
            device=device,
            accelerator=accelerator
        )
        model       = checkpoint_data['model'] if 'model' in checkpoint_data else model
        optimizer   = checkpoint_data['optimizer'] if 'optimizer' in checkpoint_data else optimizer
        start_epoch = checkpoint_data['start_epoch'] if 'start_epoch' in checkpoint_data else start_epoch
        if 'best_criterion_val' in checkpoint_data:
            best_criterion_val = checkpoint_data['best_criterion_val']
        if 'criterion' in checkpoint_data:
            config.train.best_model_criterion.criterion = checkpoint_data['criterion'] 

    if config.ddp:
        device = accelerator.device
        model, optimizer, train_loader = accelerator.prepare(
            model, optimizer, train_loader
        )
        if valid_loader is not None:
            valid_loader = accelerator.prepare(valid_loader)
    else:
        model = model.to(device)
    
    if is_main_process(accelerator):
        exp_trackers = get_experiment_trackers(config)

    trainer = get_trainer(
        config,
        model=model,
        optimizer=optimizer,
        device=device,
        accelerator=accelerator,
        epoch=start_epoch,
        work_dir=work_dir,
        ids_to_labels=data_info_train.train.ids_to_labels
    )

    eval_save_dir = work_dir / 'eval'
    evaluator = get_evaluator(
        config,
        model=model,
        save_dir=eval_save_dir,
        device=device,
        accelerator=accelerator,
    )

    logger.info(f'Best model criterion: {config.train.best_model_criterion.criterion}')
    logger.info(f'Current best criterion value: {best_criterion_val}')
    
    start_time = time.time()
    for epoch in range(start_epoch, config.epochs + 1):
        stats_train = trainer.train_epoch(train_loader)
        if valid_loader is not None:
            stats_valid = trainer.valid_epoch(valid_loader)
        else:
            stats_valid = None

        trainer.update_epoch()

        eval_stats = {}
        for data_info in data_infos_test:       
            logger.info(f'Model evaluation on {data_info.dataset_name}')    
            eval_metrics = evaluator.evaluate(data_info)
            eval_stats[data_info.dataset_name] = eval_metrics

            logger.info(f'{data_info.dataset_name} metrics:')
            for k_metric, v_metric in eval_metrics.items():
                logger.info(f'{k_metric}: {v_metric}')

        save_ckp(
            save_paths.last_weights_path, 
            model=model, 
            epoch=epoch, 
            optimizer=optimizer, 
            best_criterion_val=best_criterion_val,
            criterion=config.train.best_model_criterion.criterion,
            accelerator=accelerator
        )

        save_ckp(
            save_paths.last_emb_weights_path, 
            model=model, 
            emb_model_only=True,
            accelerator=accelerator
        )

        if is_main_process(accelerator):
            stats = dict(
                learning_rate=optimizer.param_groups[-1]['lr'],
                epoch=epoch,
                train=stats_train,
                valid=stats_valid,
                eval=eval_stats
            )
            
            is_best, current_criterion_val = is_model_best(
                stats,
                best_criterion_val,
                criterion=config.train.best_model_criterion.criterion,
                criterion_type=config.train.best_model_criterion.type
            )

            if is_best:
                best_criterion_val = current_criterion_val
                shutil.copyfile(
                    save_paths.last_weights_path, 
                    save_paths.best_weights_path
                )
                shutil.copyfile(
                    save_paths.last_emb_weights_path, 
                    save_paths.best_emb_weights_path
                )
                logger.info(f'Best model was saved based on {config.train.best_model_criterion.criterion}')

            for exp_tracker_name, exp_tracker in exp_trackers.items():
                exp_tracker.send_stats(stats)
            
            logger.info_epoch_train(
                epoch, 
                stats_train,
                stats_valid
            )
            logger.info_epoch_time(
                start_time, 
                start_epoch, 
                epoch, 
                num_epochs=config.epochs, 
                workdir_path=work_dir
            )

    if is_main_process(accelerator):
        for exp_tracker_name, exp_tracker in exp_trackers.items():
            logger.info(f'Finish {exp_tracker_name}')
            exp_tracker.finish_run()

        logger.info(f"Training done, all results were saved in {work_dir}")


if __name__=="__main__":
    train()