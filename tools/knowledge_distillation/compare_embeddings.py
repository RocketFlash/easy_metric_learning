import sys

sys.path.append("./")

import os
import argparse
import numpy as np
from pathlib import Path
import pandas as pd
import random
from omegaconf import OmegaConf

from src.utils import get_sample, get_images_paths
from src.model import load_emb_model_and_weights
from src.transform import get_transform
from src.utils import get_device
from tqdm import tqdm
import torch
from torch.nn.functional import cosine_similarity


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--work_folder_s",
        type=str,
        default="",
        help="path to student model working directory",
    )
    parser.add_argument(
        "--work_folder_t",
        type=str,
        default="",
        help="path to teacher model working directory",
    )
    parser.add_argument("--image_path", type=str, default="", help="image path")
    parser.add_argument("--dataset_path", default="", help="path to the dataset")
    parser.add_argument(
        "--n_vals", type=int, default=8, help="number of elements to show"
    )
    parser.add_argument(
        "--n_samples", type=int, default=200, help="number of samples to compare"
    )
    return parser.parse_args()


def get_model_from_work_folder(work_folder):
    work_folder = Path(work_folder)
    for config_name in (
        "config_train.yaml",
        "config_test.yaml",
        "config.yaml",
        "config.yml",
    ):
        config_path = work_folder / config_name
        if config_path.is_file():
            break
    else:
        raise FileNotFoundError(f"No config file found in {work_folder}")

    config = OmegaConf.load(config_path)
    device = get_device(config.device if "device" in config else "cpu")

    for weights_name in ("best_emb.pt", "last_emb.pt"):
        weights = work_folder / weights_name
        if weights.is_file():
            break
    else:
        raise FileNotFoundError(f"No embedding weights found in {work_folder}")

    model = (
        load_emb_model_and_weights(
            config_backbone=config.backbone,
            config_head=config.head,
            weights=weights,
            device=device,
        )
        .to(device)
        .eval()
    )
    transform = get_transform(config.transform.test)

    return model, transform, device


if __name__ == "__main__":
    args = parse_args()

    model_s, transform_s, device_s = get_model_from_work_folder(args.work_folder_s)
    model_t, transform_t, device_t = get_model_from_work_folder(args.work_folder_t)

    if args.dataset_path:
        dataset_path = Path(args.dataset_path)
        images_paths = get_images_paths(dataset_path)
        random.shuffle(images_paths)
        images_paths = images_paths[: args.n_samples]

        cos_sim_vals = []
        for image_path in tqdm(images_paths):
            sample_s = get_sample(str(image_path), transform=transform_s)
            sample_t = get_sample(str(image_path), transform=transform_t)
            with torch.no_grad():
                o_s = model_s(sample_s.to(device_s)).cpu()
                o_t = model_t(sample_t.to(device_t)).cpu()
                cos_sim = cosine_similarity(o_s, o_t)
                cos_sim_vals.append(cos_sim)
        np.set_printoptions(linewidth=200)
        print("student : ", np.round(o_s[:, : args.n_vals].numpy(), 4))
        print("teacher : ", np.round(o_t[:, : args.n_vals].numpy(), 4))
        print(f"avg cosine similarity: {sum(cos_sim_vals)/len(cos_sim_vals)}")

    if args.image_path:
        sample_s = get_sample(args.image_path, transform=transform_s)
        sample_t = get_sample(args.image_path, transform=transform_t)

        with torch.no_grad():
            o_s = model_s(sample_s.to(device_s)).cpu()
            o_t = model_t(sample_t.to(device_t)).cpu()

            np.set_printoptions(linewidth=200)
            print("student : ", np.round(o_s[:, : args.n_vals].numpy(), 4))
            print("teacher : ", np.round(o_t[:, : args.n_vals].numpy(), 4))
            print(f"cosine similarity: {cosine_similarity(o_s, o_t)}")
