import os
import argparse
import numpy as np
from pathlib import Path

from src.utils import get_train_val_split, load_ckp, load_config
from src.dataloader import TilesDataset
from src.model import get_model_embeddings
from src.transforms import get_transformations
from src.utils import Logger, get_image, plot_tiles_similarity, show_images

import pandas as pd
import torch
from torch.utils.data import DataLoader
import cv2
from tqdm import tqdm
from os.path import isfile, join
import json
from src.infer import infer, generate_embeddings
from sklearn.metrics.pairwise import cosine_similarity


def get_tiles_embeddings(model, data_loader, device):
    target_embeddings = []
    target_coordinates = []
    
    for tiles_batch, coords_batch in data_loader:
        with torch.no_grad():
            tiles_batch = tiles_batch.to(device) 
            output_embeddings = model(tiles_batch).cpu().numpy() 
            target_embeddings.append(output_embeddings)
            target_coordinates.append(coords_batch)

    target_embeddings = np.concatenate(target_embeddings, axis=0)
    target_coordinates = np.concatenate(target_coordinates, axis=0)
    return target_embeddings, target_coordinates


def main(CONFIGS, source, window_size=20, step_size=10, is_horizontal=True):

    o_s = 0.5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform_test = get_transformations('test_aug', image_size=(CONFIGS['DATA']['IMG_SIZE'],
                                                                 CONFIGS['DATA']['IMG_SIZE']))                        

    model = get_model_embeddings(model_name=CONFIGS['MODEL']['ENCODER_NAME'], 
                                 embeddings_size=CONFIGS['MODEL']['EMBEDDINGS_SIZE'],   
                                 dropout=CONFIGS['TRAIN']['DROPOUT_PROB'])

    model = load_ckp(CONFIGS['TEST']['WEIGHTS'], model, emb_model_only=True)
    model.to(device)
    model.eval()

    images_paths = sorted([l for l in list(source.glob('*.jpeg')) + \
                                      list(source.glob('*.jpg')) + \
                                      list(source.glob('*.png'))])

    for im_i in tqdm(range(len(images_paths)-1)):
        source_image_path = images_paths[im_i] 
        target_image_path = images_paths[im_i+1] 

        source_image = get_image(source_image_path)
        target_image = get_image(target_image_path)


        # Source image processing
        source_dataset = TilesDataset(source_image,
                                        window_size=window_size,
                                        step_size=step_size,
                                        is_horizontal=is_horizontal,   
                                        transform=transform_test)
        source_data_loader = DataLoader(source_dataset, batch_size=50, pin_memory=True, num_workers=8)

        # Target image processing
        target_dataset = TilesDataset(target_image,
                                        window_size=window_size,
                                        step_size=step_size,
                                        is_horizontal=is_horizontal,   
                                        transform=transform_test)
        target_data_loader = DataLoader(target_dataset, batch_size=50, pin_memory=True, num_workers=8)

        source_embeddings, source_coordinates = get_tiles_embeddings(model, source_data_loader, device)
        target_embeddings, target_coordinates = get_tiles_embeddings(model, target_data_loader, device)
        


        cosine_sim_matrix = cosine_similarity(source_embeddings, target_embeddings)
        max_similar = np.unravel_index(np.argmax(cosine_sim_matrix), cosine_sim_matrix.shape)

        source_coords_best = source_coordinates[max_similar[0]]
        target_coords_best = target_coordinates[max_similar[1]]

        source_image_small = cv2.resize(source_image, (0,0), fx=o_s, fy=o_s) 
        target_image_small = cv2.resize(target_image, (0,0), fx=o_s, fy=o_s) 
        
        source_coords = [(source_coords_best[0],source_coords_best[1]), (source_coords_best[0]+source_coords_best[2],source_coords_best[1]+source_coords_best[3])]
        target_coords = [(target_coords_best[0],target_coords_best[1]), (target_coords_best[0]+target_coords_best[2],target_coords_best[1]+target_coords_best[3])]

        source_coords = [(int(x*o_s), int(y*o_s)) for x,y in source_coords]
        target_coords = [(int(x*o_s), int(y*o_s)) for x,y in target_coords]

        stitched_image = np.concatenate([target_image_small[:, :target_coords[0][0],:], source_image_small[:, source_coords[0][0]:, :]], axis=1)
        
        cv2.rectangle(source_image_small,source_coords[0],source_coords[1],(0,255,0),5)
        cv2.rectangle(target_image_small,target_coords[0],target_coords[1],(255,0,0),5)


        show_images([target_image_small, source_image_small, stitched_image], n_col=3, save_name=CONFIGS["MISC"]["TMP"]/f'coords_{im_i}.png')
        plot_tiles_similarity(cosine_sim_matrix, save_path=CONFIGS["MISC"]["TMP"]/f'{im_i}.png')
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='', help='path to cfg.yaml')
    parser.add_argument('--model', required=True, help='path to the pretrained model')
    parser.add_argument('--source', type=str, default='', help='images folder path')
    parser.add_argument('--window_size', type=int, default=20, help='tiles window size')
    parser.add_argument('--step_size', type=int, default=10, help='tiles step size')
    parser.add_argument('--horizontal', action='store_true', help='search horizontally')
    parser.add_argument('--tmp', default="", help='tmp')
    args = parser.parse_args()

    assert os.path.isfile(args.config)
    CONFIGS = load_config(args.config)
    source = Path(args.source)

    if args.model:
        if isfile(args.model):
            CONFIGS['TEST']['WEIGHTS'] = args.model
        else:
            print("no pretrained model found at '{}'".format(args.model))

    if args.tmp != "" and args.tmp != CONFIGS["MISC"]["TMP"]:
        CONFIGS["MISC"]["TMP"] = Path(args.tmp)
        CONFIGS["MISC"]["TMP"].mkdir(exist_ok=True)

    main(CONFIGS, source, window_size=args.window_size, step_size=args.step_size, is_horizontal=args.horizontal)