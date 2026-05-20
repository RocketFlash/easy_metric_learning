import pytest

pytest.importorskip("hydra")
pytest.importorskip("omegaconf")

from src.config import ConfigValidationError
from tools.validate_config import validate_config


def test_validate_config_tool_accepts_default_train_config():
    config = validate_config(
        "config_train",
        overrides=["use_wandb=False", "use_mlflow=False"],
    )

    assert config.run_name


@pytest.mark.parametrize("loss_name", ["focal_loss", "soft_cross_entropy"])
def test_validate_config_tool_instantiates_loss_configs(loss_name):
    config = validate_config(
        "config_train",
        overrides=[
            f"loss={loss_name}",
            "n_classes=3",
            "use_wandb=False",
            "use_mlflow=False",
        ],
    )

    assert config.loss.losses


def test_validate_config_tool_surfaces_training_validation_errors():
    with pytest.raises(ConfigValidationError, match="fsdp"):
        validate_config(
            "config_train",
            overrides=[
                "use_wandb=False",
                "use_mlflow=False",
                "ddp=False",
                "train.trainer.fsdp.enabled=True",
            ],
        )
