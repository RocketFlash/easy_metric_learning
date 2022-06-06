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
        source_tile = source_image[:, :window_size] if is_horizontal else source_image[window_size:window_size*2, :]

        sample = transform_test(image=source_tile)
        source_tile = sample['image']
        source_tile_torch = source_tile.unsqueeze(0)

        with torch.no_grad():
            source_tile_torch = source_tile_torch.to(device=device) 
            source_tile_embeddings = model(source_tile_torch).cpu().numpy()

        # Target image processing
        dataset = TilesDataset(target_image,
                               window_size=window_size,
                               step_size=step_size,
                               is_horizontal=is_horizontal,   
                               transform=transform_test)
        data_loader = DataLoader(dataset, batch_size=50, pin_memory=True, num_workers=8)

        all_embeddings = []
        all_coordinates = []
        for tiles_batch, coords_batch in data_loader:
            with torch.no_grad():
                tiles_batch = tiles_batch.to(device) 
                output_embeddings = model(tiles_batch).cpu().numpy() 
                all_embeddings.append(output_embeddings)
                all_coordinates.append(coords_batch)
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_coordinates = np.concatenate(all_coordinates, axis=0)
        cosine_sim_matrix = cosine_similarity(source_tile_embeddings, all_embeddings)
        
        max_similar = np.argmax(cosine_sim_matrix[0])
        coords_best = all_coordinates[max_similar]

        source_image_small = cv2.resize(source_image, (0,0), fx=o_s, fy=o_s) 
        target_image_small = cv2.resize(target_image, (0,0), fx=o_s, fy=o_s) 
        
        s_h, s_w, s_c = source_image.shape
        source_coords = [(0, 0), (window_size, s_h)] if is_horizontal else [(0, window_size), (s_w, 2*window_size)]
        target_coords = [(coords_best[0],coords_best[1]), (coords_best[0]+coords_best[2],coords_best[1]+coords_best[3])]

        source_coords = [(int(x*o_s), int(y*o_s)) for x,y in source_coords]
        target_coords = [(int(x*o_s), int(y*o_s)) for x,y in target_coords]

        cv2.rectangle(source_image_small,source_coords[0],source_coords[1],(0,255,0),5)
        cv2.rectangle(target_image_small,target_coords[0],target_coords[1],(255,0,0),5)

        show_images([target_image_small, source_image_small], n_col=2, save_name=CONFIGS["MISC"]["TMP"]/f'coords_{im_i}.png')
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