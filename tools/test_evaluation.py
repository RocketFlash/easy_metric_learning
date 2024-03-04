import sys
sys.path.append("./")

import hydra
from pathlib import Path

from src.model import get_model
from src.utils import seed_everything, get_device
from src.data import get_test_data_from_config
from src.evaluator import get_evaluator


@hydra.main(version_base=None,
            config_path='../configs/',
            config_name='config')
def test_dataloader(config):
    seed_everything(config.random_state)

    work_dir = Path(config.work_dirs) / config.run_name
    work_dir.mkdir(exist_ok=True)
    
    data_infos_test = get_test_data_from_config(config)

    device = get_device(config.device)
    model = get_model(
        config_backbone=config.backbone,
        config_head=config.head,
    ).to(device)

    evaluator = get_evaluator(
        config,
        model=model,
        work_dir=work_dir,
        device=device,
    )
    
    for data_info in data_infos_test:
        metrics = evaluator.evaluate(data_info)
        print(metrics)


if __name__ == '__main__':
    test_dataloader()