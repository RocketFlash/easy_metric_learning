import sys
sys.path.append("./")

import os
import argparse
import numpy as np
from pathlib import Path
import pandas as pd
import random

from src.utils import (load_ckp, 
                       load_config, 
                       get_sample, 
                       get_images_paths)
from src.dataset import get_loader
from src.model import get_model_embeddings
from tqdm import tqdm
import torch
from torch.nn.functional import cosine_similarity


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_folder_s', type=str, default='', help='path to student model working directory')
    parser.add_argument('--work_folder_t', type=str, default='', help='path to teacher model working directory')
    parser.add_argument('--image_path', type=str, default='', help='image path')
    parser.add_argument('--dataset_path', default='', help='path to the dataset')
    parser.add_argument('--n_vals', type=int, default=8, help='number of elements to show')
    parser.add_argument('--n_samples', type=int, default=200, help='number of samples to compare')
    return parser.parse_args()


def get_model_from_work_folder(work_folder):
    work_folder = Path(work_folder)
    config = work_folder / 'config.yml'

    assert os.path.isfile(config)
    CONFIGS = load_config(config)

    img_size = CONFIGS['DATA']['IMG_SIZE']
    device = torch.device(CONFIGS['GENERAL']['DEVICE'])
    weights = work_folder / 'best_emb.pt'

    model = get_model_embeddings(model_config=CONFIGS['MODEL'])
    model = load_ckp(weights, model, emb_model_only=True)
    model.to(device)
    model.eval()

    return model, img_size, device


if __name__ == '__main__':
    args = parse_args()

    model_s, img_size_s, device_s = get_model_from_work_folder(args.work_folder_s)
    model_t, img_size_t, device_t = get_model_from_work_folder(args.work_folder_t)

    if args.dataset_path:
        dataset_path = Path(args.dataset_path)
        images_paths = get_images_paths(dataset_path)
        random.shuffle(images_paths)
        images_paths = images_paths[:args.n_samples]

        cos_sim_vals = []
        for image_path in tqdm(images_paths):
            sample_s = get_sample(str(image_path),
                            img_h=img_size_s,
                            img_w=img_size_s)
            sample_t = get_sample(str(image_path),
                                img_h=img_size_t,
                                img_w=img_size_t)
            with torch.no_grad():
                o_s = model_s(sample_s.to(device_s)).cpu()
                o_t = model_t(sample_t.to(device_t)).cpu()
                cos_sim = cosine_similarity(o_s, o_t)
                cos_sim_vals.append(cos_sim)
        np.set_printoptions(linewidth=200)
        print('student : ', np.round(o_s[:, :args.n_vals].numpy(), 4))
        print('teacher : ', np.round(o_t[:, :args.n_vals].numpy(), 4))
        print(f'avg cosine similarity: {sum(cos_sim_vals)/len(cos_sim_vals)}')
            

    if args.image_path:
        sample_s = get_sample(args.image_path,
                            img_h=img_size_s,
                            img_w=img_size_s)
        sample_t = get_sample(args.image_path,
                            img_h=img_size_t,
                            img_w=img_size_t)

        with torch.no_grad():
            o_s = model_s(sample_s.to(device_s)).cpu()
            o_t = model_t(sample_t.to(device_t)).cpu()

            np.set_printoptions(linewidth=200)
            print('student : ', np.round(o_s[:, :args.n_vals].numpy(), 4))
            print('teacher : ', np.round(o_t[:, :args.n_vals].numpy(), 4))
            print(f'cosine similarity: {cosine_similarity(o_s, o_t)}')

    