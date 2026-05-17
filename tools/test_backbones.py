import sys

sys.path.append("./")

import hydra
import torch
from src.model import get_model


@hydra.main(version_base=None, config_path="../configs/", config_name="config_train")
def test_backbone(config):
    n_classes = 2828
    image_size = 224
    # margin_config = None
    margin_config = config.margin
    model = get_model(
        config_backbone=config.backbone,
        config_head=config.head,
        config_margin=margin_config,
        n_classes=n_classes,
    ).eval()

    sample = torch.rand((8, 3, image_size, image_size))

    if margin_config is not None:
        label = torch.zeros(sample.shape[0]).int()
        output = model(sample, label)
    else:
        output = model(sample)

    print(model)
    if isinstance(output, tuple):
        print(output[0].shape)
    else:
        print(output.shape)


if __name__ == "__main__":
    test_backbone()
