from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torchvision")
pytest.importorskip("hydra")
pytest.importorskip("omegaconf")

from hydra import compose, initialize_config_dir

from src.transform import get_transform
from src.transform.torchvision_policy import RandomErasing, TorchvisionPolicy

CONFIG_DIR = str((Path(__file__).resolve().parents[1] / "configs").resolve())


def test_torchvision_policy_preserves_image_shape_and_dtype():
    transform = TorchvisionPolicy(policy="randaugment", p=1.0)
    image = np.full((16, 16, 3), 128, dtype=np.uint8)

    transformed = transform(image=image)["image"]

    assert transformed.shape == image.shape
    assert transformed.dtype == np.uint8


def test_random_erasing_masks_part_of_image():
    transform = RandomErasing(
        scale=(0.25, 0.25),
        ratio=(1.0, 1.0),
        fill=0,
        p=1.0,
    )
    image = np.full((20, 20, 3), 255, dtype=np.uint8)

    transformed = transform(image=image)["image"]

    assert transformed.shape == image.shape
    assert np.any(transformed == 0)
    assert np.any(transformed == 255)


def test_default_albumentations_policy_instantiates():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="config_train",
            overrides=["transform=default", "img_h=48", "img_w=48"],
        )

    transform = get_transform(cfg.transform.train)
    transformed = transform(image=np.full((48, 48, 3), 128, dtype=np.uint8))

    assert transformed["image"].shape[-2:] == (48, 48)
