import sys
sys.path.append("./")

import os
import argparse
import numpy as np
from pathlib import Path
import pandas as pd

from src.utils import load_ckp, load_config
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
    emb_size = args.emb_size
    n_workers=args.n_jobs

    if dataset_path:
        CONFIGS["DATA"]["DIR"] = dataset_path

    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True)

    device = torch.device(CONFIGS['GENERAL']['DEVICE'])

    df_dtype = {'label': str,
                'file_name': str,
                'width': int,
                'height': int,
                'is_test':int}  

    df = pd.read_csv(dataset_csv, 
                     dtype=df_dtype)
    
    if args.dataset_type in ['cars', 'sop']:
        df = df[df['is_test'] == 1].reset_index()
    
    CONFIGS['DATA']['USE_CATEGORIES'] = False
    data_loader, dataset = get_loader(df,
                                      data_config=CONFIGS["DATA"],
                                      split='val',
                                      test=True,
                                      batch_size=bs,
                                      num_thread=n_workers,
                                      label_column='label',
                                      fname_column='file_name',
                                      return_filenames=True,
                                      use_bboxes=args.use_bboxes)
    ids_to_labels = dataset.get_ids_to_labels()

    model = get_model_embeddings(model_config=CONFIGS['MODEL'])
    model = load_ckp(weights, model, emb_model_only=True)
    model.to(device)

    embeddings = np.zeros((len(df), emb_size), dtype=np.float32)
    labels = np.zeros(len(df), dtype=object)
    file_names = np.zeros(len(df), dtype=object)

    tqdm_bar = tqdm(data_loader, total=int(len(data_loader)))

    model.eval()
    with torch.no_grad():
        index = 0
        for batch_index, (data, targets, file_nms) in enumerate(tqdm_bar):
            data = data.to(device)

            output = model(data)
            batch_size = output.shape[0]

            lbls = [ids_to_labels[t] for t in targets.cpu().numpy()]
            embeddings[index:(index+batch_size), :] = output.cpu().numpy()
            labels[index:(index+batch_size)] = lbls
            file_names[index:(index+batch_size)] = file_nms
            index += batch_size
    
    non_empty_rows_mask = file_names != ''
    embeddings = embeddings[non_empty_rows_mask]
    labels = labels[non_empty_rows_mask]
    file_names = file_names[non_empty_rows_mask]

    np.savez(save_path / 'embeddings.npz', 
             embeddings=embeddings, 
             labels=labels,
             file_names=file_names)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_folder', type=str, default='', help='path to trained model working directory')
    parser.add_argument('--config', type=str, default='', help='path to cfg.yaml')
    parser.add_argument('--weights', type=str, default='', help='weights path')
    parser.add_argument('--save_path', type=str, default='', help='save path')
    parser.add_argument('--dataset_path', type=str, default='', help='path to dataset')
    parser.add_argument('--dataset_csv', type=str, default='', help='path to dataset csv file')
    parser.add_argument('--dataset_type', type=str, default='general', help='dataset type one of [general, cars, sop]')
    parser.add_argument('--bs',type=int, default=8, help='batch size')
    parser.add_argument('--n_jobs', type=int, default=4, help='number of parallel jobs')
    parser.add_argument('--emb_size',type=int, default=512, help='embeddings size')
    parser.add_argument('--use_bboxes', action='store_true', help='use regions from bboxes')
    args = parser.parse_args()

    if args.work_folder:
        args.work_folder = Path(args.work_folder)
        args.config = args.work_folder / 'config.yml'
        args.weights = args.work_folder / 'best_emb.pt'
        dataset_name = Path(args.dataset_path).name
        args.save_path = args.work_folder / 'embeddings' / dataset_name

    assert os.path.isfile(args.config)
    CONFIGS = load_config(args.config)

    save_path = Path(args.save_path)
    save_path.mkdir(exist_ok=True, parents=True)

    main(CONFIGS, args)