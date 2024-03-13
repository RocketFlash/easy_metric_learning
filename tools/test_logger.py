import sys
sys.path.append("./")

import hydra
from pathlib import Path
from src.logger import Logger
from src.data import get_train_data_from_config
from src.model import get_model
from src.utils import get_device



@hydra.main(version_base=None,
            config_path='../configs/',
            config_name='config')
def test_dataloader(config):
    work_dir = Path(config.work_dirs) / config.run_name
    work_dir.mkdir(exist_ok=True)

    logger = Logger(work_dir / "log.txt")
    logger.info_config(config)

    data_info_train = get_train_data_from_config(config)

    device = get_device(config.device)
    model = get_model(
        config_backbone=config.backbone,
        config_head=config.head,
        config_margin=config.margin,
        n_classes=data_info_train.train.dataset_stats.n_classes
    ).to(device)

    for i in range(5):
        logger.info(i)
        logger.info_model(config)



if __name__ == '__main__':
    test_dataloader()