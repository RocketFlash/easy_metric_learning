import albumentations as A
from .cutmix import cutmix
from .mixup import mixup
import hydra
from torchvision import transforms


def get_inverse_transfrom(mean, std):
    inv_std  = [1/x for x in std]
    inv_mean = [-x for x in mean]

    return transforms.Compose(
        [
            transforms.Normalize(
                mean=[ 0., 0., 0. ],
                std=inv_std
            ),
            transforms.Normalize(
                mean=inv_mean,
                std=[ 1., 1., 1. ]
            ),
        ]
    )

def get_transform(transform_config):
    transforms = []

    for augmentation_name in transform_config.get("order"):
        transform = hydra.utils.instantiate(
            transform_config.get(augmentation_name), 
            _convert_="object"
        )
        transforms.append(transform)

    return A.Compose(transforms)