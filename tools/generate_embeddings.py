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

def main(CONFIGS, args):

    weights = args.weights
    dataset_csv = args.dataset_csv
    dataset_path = args.dataset_path
    save_path = args.save_path 
    bs = args.bs

    if dataset_path:
        CONFIGS["DATA"]["DIR"] = dataset_path

    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True)

    device = torch.device(CONFIGS['GENERAL']['DEVICE'])

    df = pd.read_csv(dataset_csv, dtype={'label': str,
                                         'file_name': str,
                                         'width': int,
                                         'height': int,
                                         'label_id': int})
    
    data_loader, dataset = get_loader(df,
                              data_config=CONFIGS["DATA"],
                              split='val',
                              batch_size=bs,
                              return_filenames=True)

    model = get_model_embeddings(model_config=CONFIGS['MODEL'])

    model = load_ckp(weights, model, emb_model_only=True)
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
                labels.append(int(label.cpu()))

    embeddings = np.array(embeddings)
    file_names = np.array(file_names)
    labels = np.array(labels)

    np.savez(save_path / 'embeddings_nontrain.npz', 
             embeddings=embeddings, 
             labels=labels,
             file_names=file_names)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='', help='path to cfg.yaml')
    parser.add_argument('--weights', type=str, default='', help='weights path')
    parser.add_argument('--save_path', type=str, default='', help='save path')
    parser.add_argument('--dataset_path', type=str, default='', help='path to dataset')
    parser.add_argument('--dataset_csv', type=str, default='', help='path to dataset csv file')
    parser.add_argument('--bs',type=int, default=8, help='batch size')
    args = parser.parse_args()

    assert os.path.isfile(args.config)
    CONFIGS = load_config(args.config)

    save_path = Path(args.save_path)
    save_path.mkdir(exist_ok=True)

    main(CONFIGS, args)