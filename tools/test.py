import sys
sys.path.append("./")

import torch
import hydra
from pathlib import Path
import multiprocessing
from omegaconf import OmegaConf

from src.logger import Logger
from src.data import get_test_data_from_config
from src.model import get_model
from src.utils import (load_checkpoint,
                       load_model_except_torch,
                       seed_everything, 
                       get_device)
from src.evaluator import get_evaluator


@hydra.main(version_base=None,
            config_path='../configs/',
            config_name='config_test')
def test(config):
    seed_everything(config.random_state)

    if config.n_workers=='auto':
        config.n_workers = multiprocessing.cpu_count()

    work_dir = Path(config.work_dirs) / config.run_name
    work_dir.mkdir(exist_ok=True, parents=True)

    logger = Logger(work_dir / "log_test.txt")
    if config.debug: logger.info('DEBUG MODE')
    OmegaConf.save(config, work_dir / "config_test.yaml")
    
    data_infos_test = get_test_data_from_config(config)
    device = get_device(config.device)

    if config.model_type=='torch':
        model = get_model(
            config_backbone=config.backbone,
            config_head=config.head,
        ).to(device)

        if config.head.type=='no_head':
            config.embeddings_size = model.backbone_out_feats
            logger.info(f'Embeddings size changed to backbone output size {config.embeddings_size}')

        logger.info_model(config)

        if config.weights is not None:
            checkpoint_data = load_checkpoint(
                config.weights, 
                model=model, 
                logger=logger, 
                mode='emb'
            )
            model = checkpoint_data['model'] if 'model' in checkpoint_data else model
        model_info = {
            'model' : model,
            'model_type' : 'torch'
        }
    else:
        model_info = load_model_except_torch(
            config.weights, 
            model_type=config.model_type, 
            device=device,
            logger=logger
        )
        model = model_info['model']
    
    eval_save_dir = work_dir / 'test'
    evaluator = get_evaluator(
        config,
        save_dir=eval_save_dir ,
        device=device,
        model_info=model_info
    )
    
    for data_info in data_infos_test:       
        logger.info(f'Model evaluation on {data_info.dataset_name}')    
        eval_metrics = evaluator.evaluate(data_info)

        logger.info(f'{data_info.dataset_name} metrics:')
        for k_metric, v_metric in eval_metrics.items():
            logger.info(f'{k_metric}: {v_metric}')


if __name__=="__main__":
    test()