import os
import argparse
import numpy as np
from pathlib import Path

from src.infer import generate_embeddings
from src.utils import get_train_val_split, load_ckp, load_config
from src.dataloader import get_loader
from src.model import get_model_embeddings

import torch

def main(CONFIGS, bs=8, train_embeddings=False, save_separate=False):
    device = torch.device(CONFIGS['GENERAL']['DEVICE'])
    df_train, df_valid, df_full = get_train_val_split(CONFIGS["DATA"]["SPLIT_FILE"],
                                                      fold=CONFIGS["DATA"]["FOLD"])
    df = df_train if train_embeddings else df_valid
    valid_loader = get_loader(CONFIGS["DATA"]["DIR"],
                              df,
                              batch_size=bs,
                              split='val',
                              num_thread=CONFIGS["DATA"]["WORKERS"], 
                              img_size=CONFIGS['DATA']['IMG_SIZE'])
    model = get_model_embeddings(model_name=CONFIGS['MODEL']['ENCODER_NAME'], 
                                 embeddings_size=CONFIGS['MODEL']['EMBEDDINGS_SIZE'],   
                                 dropout=CONFIGS['TRAIN']['DROPOUT_PROB'])

    model = load_ckp(CONFIGS['TEST']['WEIGHTS'], model, emb_model_only=True)
    model.to(device)

    embeddings = generate_embeddings(model, 
                                     dataloader=valid_loader,  
                                     device=device)
    
    save_name_labels = "train_labels.npy" if train_embeddings else "labels.npy"
    save_name_embeddings = "train_embeddings.npy" if train_embeddings else "embeddings.npy"
    save_name_full = "train_embeddings_labels.npy" if train_embeddings else "embeddings_labels.npy"

    np.save(Path(CONFIGS['TEST']['EMBEDDINGS_SAVE_PATH'])/save_name_full, embeddings)

    if save_separate:
        np.save(Path(CONFIGS['TEST']['EMBEDDINGS_SAVE_PATH'])/save_name_labels, embeddings['labels'])
        np.save(Path(CONFIGS['TEST']['EMBEDDINGS_SAVE_PATH'])/save_name_embeddings, embeddings['embeddings'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='', help='path to cfg.yaml')
    parser.add_argument('--bs',type=int, default=8, help='batch size')
    parser.add_argument('--train_embeddings', action='store_true', help='generate embeddings from training data')
    parser.add_argument('--save_separate', action='store_true', help='save labels and embeddings separately')
    parser.add_argument('--tmp', default="", help='tmp')
    args = parser.parse_args()

    assert os.path.isfile(args.config)
    CONFIGS = load_config(args.config)

    main(CONFIGS, args.bs, args.train_embeddings, args.save_separate)