import argparse
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np
from utils import (add_image_sizes, 
                   get_labels_and_paths,
                   download_dataset)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', 
                        type=str, 
                        default='./', 
                        help='dataset path')
    parser.add_argument('--save_path', 
                        type=str, 
                        default='./', 
                        help='save path')
    parser.add_argument('--download', 
                        action='store_true', 
                        help='Dowload images')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.download:
        save_path = Path(args.save_path)
        save_path.mkdir(exist_ok=True)
        download_dataset(save_path, dataset='largefinefoodai')
        dataset_path = save_path / 'largefinefoodai'
    else:
        dataset_path = Path(args.dataset_path)
    save_path = dataset_path

    # train_path = dataset_path / 'train'
    # val_path   = dataset_path / 'val'

    # df_train = get_labels_and_paths(train_path, split='train')
    # df_val   = get_labels_and_paths(val_path, split='val')

    # df_train['is_test'] = 0
    # df_val['is_test']   = 1

    # df_info = pd.concat([df_train, df_val],
    #                     ignore_index=True)
    
    # df_info['label'] = df_info['label'].apply(lambda x: f'ali_{x}')
    
    # df_info = add_image_sizes(df_info, dataset_path)
    # df_info = df_info[['file_name', 
    #                    'label', 
    #                    'width', 
    #                    'height',
    #                    'is_test']]
    # df_info.to_csv(save_path / 'dataset_info.csv', index=False) 

    