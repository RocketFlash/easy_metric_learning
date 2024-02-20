import sys
sys.path.append("./")

import hydra
import torch
from src.data.utils import (get_train_val_split,
                            get_object_from_omegaconf)

@hydra.main(version_base=None,
            config_path='../configs/',
            config_name='config')
def test_dataloader(config):

    annotations = get_object_from_omegaconf(config.dataset.annotations)
    root_dir = get_object_from_omegaconf(config.dataset.dir)

    df_train, df_valid, df_full = get_train_val_split(
        annotation=annotations, 
        fold=config.dataset.fold
    )


    # n_classes = 2828
    # image_size = 224
    # # margin_config = None
    # margin_config = config.margin
    # model = get_model(
    #     config.backbone,
    #     config.head,
    #     margin_config=margin_config,
    #     n_classes=n_classes
    # ).eval()

    # sample = torch.rand((8, 3, image_size, image_size))

    # if margin_config is not None:
    #     label = torch.zeros(sample.shape[0]).int()
    #     output = model(sample, label)
    # else:
    #     output = model(sample)

    # print(model)
    # print(output.shape)

if __name__ == '__main__':
    test_dataloader()