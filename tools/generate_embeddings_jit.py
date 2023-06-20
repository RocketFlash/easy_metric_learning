import sys
sys.path.append("./")

import os
import argparse
import numpy as np
from pathlib import Path
import pandas as pd

from src.infer import generate_embeddings, get_embeddings
from src.utils import get_train_val_split, load_ckp, load_config
from src.dataset import get_loader
from src.model import get_model_embeddings

from tqdm import tqdm
import torch

def main(args):
    save_path = Path(args.save_path)
    save_path.mkdir(exist_ok=True)

    weights = args.weights
    dataset_csv = args.dataset_csv
    dataset_path = args.dataset_path
    save_path = args.save_path 
    bs = args.bs

    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True)

    device = torch.device(args.device)

    df = pd.read_csv(dataset_csv, dtype={
                                         'label': str,
                                         'file_name': str,
                                         'width': int,
                                         'height': int,
                                         'label_id': int
                                         })
    
    data_loader, dataset = get_loader(df,
                              root_dir=dataset_path,
                              split='val',
                              batch_size=bs,
                              label_column='label',
                              fname_column='file_name',
                              return_filenames=True)
    
    model = torch.jit.load(args.weights, map_location=device)
    model.to(device)


    embeddings = []
    labels = []
    file_names = []

    tqdm_bar = tqdm(data_loader, total=int(len(data_loader)))

    model.eval()
    with torch.no_grad():
        for batch_index, (data, targets, file_nms) in enumerate(tqdm_bar):
            data = data.to(device)

            output = model(data)
            encodings = output.cpu().numpy()

            file_names += file_nms
            for encoding, label in zip(encodings, targets):
                embeddings.append(encoding)
                labels.append(label)

    embeddings = np.array(embeddings)
    file_names = np.array(file_names)
    labels = np.array(labels)

    np.savez(save_path / 'embeddings.npz', 
             embeddings=embeddings, 
             labels=labels,
             file_names=file_names)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='weights path')
    parser.add_argument('--save_path', type=str, default='', help='save path')
    parser.add_argument('--dataset_path', type=str, default='', help='path to dataset')
    parser.add_argument('--dataset_csv', type=str, default='', help='path to dataset csv file')
    parser.add_argument('--device', type=str, default='cpu', help='device')
    parser.add_argument('--bs',type=int, default=8, help='batch size')
    args = parser.parse_args()

    main(args)