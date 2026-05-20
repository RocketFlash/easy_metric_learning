import math
import random

import albumentations as A
import numpy as np
from PIL import Image
from torchvision import transforms as T


def _as_uint8_image(image):
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0) * 255.0
    else:
        image = np.clip(image, 0, 255)
    return image.round().astype(np.uint8)


class TorchvisionPolicy(A.ImageOnlyTransform):
    """Albumentations wrapper for torchvision policy augmentations."""

    def __init__(
        self,
        policy="randaugment",
        num_ops=2,
        magnitude=9,
        num_magnitude_bins=31,
        severity=3,
        mixture_width=3,
        chain_depth=-1,
        alpha=1.0,
        always_apply=False,
        p=0.5,
    ):
        super(TorchvisionPolicy, self).__init__(always_apply=always_apply, p=p)
        if policy == "randaugment":
            self.transform = T.RandAugment(
                num_ops=num_ops,
                magnitude=magnitude,
                num_magnitude_bins=num_magnitude_bins,
            )
        elif policy == "trivialaugment":
            self.transform = T.TrivialAugmentWide(num_magnitude_bins=num_magnitude_bins)
        elif policy == "augmix":
            self.transform = T.AugMix(
                severity=severity,
                mixture_width=mixture_width,
                chain_depth=chain_depth,
                alpha=alpha,
            )
        else:
            raise ValueError(f"Unknown torchvision policy: {policy}")

    def apply(self, image, **params):
        pil_image = Image.fromarray(_as_uint8_image(image))
        return np.asarray(self.transform(pil_image))


class RandomErasing(A.ImageOnlyTransform):
    """Random erasing for numpy images inside an albumentations pipeline."""

    def __init__(
        self,
        scale=(0.02, 0.33),
        ratio=(0.3, 3.3),
        fill=0,
        max_attempts=10,
        always_apply=False,
        p=0.5,
    ):
        super(RandomErasing, self).__init__(always_apply=always_apply, p=p)
        self.scale = tuple(scale)
        self.ratio = tuple(ratio)
        self.fill = fill
        self.max_attempts = max_attempts

    def _get_fill_value(self, image):
        if self.fill == "random":
            return np.random.randint(0, 256, size=image.shape[-1], dtype=np.uint8)
        return self.fill

    def apply(self, image, **params):
        output = image.copy()
        height, width = output.shape[:2]
        area = height * width
        log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))

        for _ in range(self.max_attempts):
            target_area = random.uniform(*self.scale) * area
            aspect_ratio = math.exp(random.uniform(*log_ratio))
            erase_height = int(round(math.sqrt(target_area * aspect_ratio)))
            erase_width = int(round(math.sqrt(target_area / aspect_ratio)))

            if erase_height < height and erase_width < width:
                top = random.randint(0, height - erase_height)
                left = random.randint(0, width - erase_width)
                output[
                    top : top + erase_height,
                    left : left + erase_width,
                    ...,
                ] = self._get_fill_value(output)
                return output

        return output
