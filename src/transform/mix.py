import numpy as np
from .cutmix import cutmix
from .mixup import mixup


def mix_transform(
        images, 
        targets,
        cutmix_p=0,
        cutmix_alpha=0.1,
        mixup_p=0,
        mixup_alpha=0.1
    ):
    p = np.random.rand()
    is_mixed = False

    if cutmix_p>0 or mixup_p>0:
        if p < cutmix_p and p < mixup_p:
            p = np.random.rand()
            if p < 0.5:
                images, targets = cutmix(images, targets, cutmix_alpha)
            else:
                images, targets = mixup(images, targets, mixup_alpha)
            is_mixed = True
        elif p < cutmix_p:
            images, targets = cutmix(images, targets, cutmix_alpha)
            is_mixed = True
        elif p < mixup_p:
            images, targets = mixup(images, targets, mixup_alpha)
            is_mixed = True

    return images, targets, is_mixed