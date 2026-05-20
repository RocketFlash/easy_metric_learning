import argparse
import sys
from pathlib import Path

sys.path.append("./")

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.config import validate_training_config
from src.loss import get_loss

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def validate_config(config_name="config_train", overrides=None):
    overrides = [] if overrides is None else list(overrides)
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        config = compose(config_name=config_name, overrides=overrides)
    OmegaConf.resolve(config)
    if config_name == "config_train":
        validate_training_config(config)
        get_loss(config.loss, device="cpu")
    return config


def parse_args():
    parser = argparse.ArgumentParser(description="Validate Hydra config composition")
    parser.add_argument("--config-name", default="config_train")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main():
    args = parse_args()
    validate_config(args.config_name, overrides=args.overrides)
    print(f"{args.config_name} is valid")


if __name__ == "__main__":
    main()
