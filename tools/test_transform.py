import sys
sys.path.append("./")

import hydra
import torch
from src.transform import get_transform


@hydra.main(version_base=None,
            config_path='../configs/',
            config_name='config')
def test_transform(config):
    transform = get_transform(config.transform.train)
    print(transform)


if __name__ == '__main__':
    test_transform()