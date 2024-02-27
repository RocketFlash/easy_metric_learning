import sys
sys.path.append("./")

import time
import shutil
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from src.data import get_data_from_config
from src.model import get_model
from src.optimizer import get_optimizer
from src.logger import Logger
from src.data.utils import save_labels_to_ids

from src.utils import load_checkpoint, save_ckp, get_save_paths
from src.utils import seed_everything, get_device
from src.trainer import MLTrainer
from src.experiment_tracker import get_experiment_trackers


@hydra.main(version_base=None,
            config_path='../configs/',
            config_name='config')
def train(config):
    seed_everything(config.random_state)

    work_dir = Path(config.work_dirs) / config.run_name
    work_dir.mkdir(exist_ok=True)

    logger = Logger(work_dir / "log.txt")
    logger.info_config(config)
    if config.debug: logger.info('DEBUG MODE')
    OmegaConf.save(config, work_dir / "config.yaml")
    
    best_loss = 10000
    start_epoch = 1
    
    save_paths = get_save_paths(work_dir)
    data_info  = get_data_from_config(config)

    train_loader = data_info.train_loader
    valid_loader = data_info.valid_loader

    save_labels_to_ids(data_info.labels_to_ids, save_dir=work_dir)
    config.margin.id_counts = data_info.dataset_stats.id_counts
    logger.info_data(data_info.dataset_stats) 
    
    device = get_device(config.device)
    model = get_model(
        config_backbone=config.backbone,
        config_head=config.head,
        config_margin=config.margin,
        n_classes=data_info.n_classes
    ).to(device)
    logger.info_model(config)

    optimizer = get_optimizer(model, optimizer_config=config.optimizer)
    
    if config.load_checkpoint is not None:
        checkpoint_data = load_checkpoint(
            config.load_checkpoint, 
            model=model, 
            optimizer=optimizer, 
            logger=logger, 
            mode=config.load_mode
        )
        model       = checkpoint_data['model'] if 'model' in checkpoint_data else model
        optimizer   = checkpoint_data['optimizer'] if 'optimizer' in checkpoint_data else optimizer
        start_epoch = checkpoint_data['start_epoch'] if 'start_epoch' in checkpoint_data else start_epoch
        best_loss   = checkpoint_data['best_loss'] if 'best_loss' in checkpoint_data else best_loss
   
    logger.info(f'Current best loss: {best_loss}')
    
    exp_trackers = get_experiment_trackers(config)

    trainer = MLTrainer(
        config,
        model=model,
        optimizer=optimizer, 
        logger=logger, 
        device=device,
        epoch=start_epoch,
        work_dir=work_dir,
        ids_to_labels=data_info.ids_to_labels
    )

    start_time = time.time()
    for epoch in range(start_epoch, config.epochs + 1):
        stats_train = trainer.train_epoch(train_loader)
        stats_valid = trainer.valid_epoch(valid_loader)
        check_loss = stats_valid.loss

        trainer.update_epoch()
        
        save_ckp(
            save_paths.last_weights_path, 
            model, 
            epoch, 
            optimizer, 
            best_loss
        )
        save_ckp(
            save_paths.last_emb_weights_path, 
            model.embeddings_net, 
            emb_model_only=True
        )
        
        if check_loss < best_loss:
            logger.info('Saving best model')
            best_loss = check_loss
            shutil.copyfile(
                save_paths.last_weights_path, 
                save_paths.best_weights_path
            )
            shutil.copyfile(
                save_paths.last_emb_weights_path, 
                save_paths.best_emb_weights_path
            )
            
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

    logger.info(f"Training done, all results saved to {work_dir}")


if __name__=="__main__":
    train()