import sys
sys.path.append("./")

import hydra
import torch
from pathlib import Path

from src.model import (get_model,
                       get_model_teacher)
from src.utils import seed_everything, get_device
# from src.data import get_test_data_from_config
# from src.evaluator import get_evaluator


@hydra.main(version_base=None,
            config_path='../configs/',
            config_name='config_train')
def test_dataloader(config):
    seed_everything(config.random_state)

    if config.distillation.teacher.model is not None:
        model_teacher = get_model_teacher(config.distillation.teacher)

        sample = torch.rand((8,3,224,224))
        print(model_teacher(sample).shape)

    
    
if __name__ == '__main__':
    test_dataloader()