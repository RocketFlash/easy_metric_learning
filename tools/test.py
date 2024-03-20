import sys
sys.path.append("./")

import hydra
from pathlib import Path
from omegaconf import OmegaConf

from src.logger import Logger
from src.data import get_test_data_from_config
from src.model import get_model
from src.utils import (load_checkpoint,
                       seed_everything, 
                       get_device)
from src.evaluator import get_evaluator


@hydra.main(version_base=None,
            config_path='../configs/',
            config_name='config')
def test(config):
    seed_everything(config.random_state)

    work_dir = Path(config.work_dirs) / config.run_name
    work_dir.mkdir(exist_ok=True)

    logger = Logger(work_dir / "log.txt")
    logger.info_config(config)
    if config.debug: logger.info('DEBUG MODE')
    OmegaConf.save(config, work_dir / "config.yaml")
    
    data_infos_test = get_test_data_from_config(config)
    
    device = get_device(config.device)
    model = get_model(
        config_backbone=config.backbone,
        config_head=config.head,
    ).to(device)
    logger.info_model(config)

    if config.load_checkpoint is not None:
        checkpoint_data = load_checkpoint(
            config.load_checkpoint, 
            model=model, 
            logger=logger, 
            mode=config.load_mode
        )
        model = checkpoint_data['model'] if 'model' in checkpoint_data else model
    
    eval_save_dir = work_dir / 'test'
    evaluator = get_evaluator(
        config,
        model=model,
        save_dir=eval_save_dir ,
        device=device,
    )
    
    for data_info in data_infos_test:       
        logger.info(f'Model evaluation on {data_info.dataset_name}')    
        eval_metrics = evaluator.evaluate(data_info)

        logger.info(f'{data_info.dataset_name} metrics:')
        for k_metric, v_metric in eval_metrics.items():
            logger.info(f'{k_metric}: {v_metric}')


if __name__=="__main__":
    test()