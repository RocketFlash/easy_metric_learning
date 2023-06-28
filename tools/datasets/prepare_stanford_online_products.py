import argparse
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np
import wget
import zipfile
from utils import add_image_sizes, download_dataset


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
        download_dataset(save_path, dataset='sop')
        dataset_path = save_path / 'Stanford_Online_Products'
    else:
        dataset_path = Path(args.dataset_path)
    save_path = dataset_path

    df_info  = pd.read_csv(save_path / 'Ebay_info.txt', sep=' ')
    df_train = pd.read_csv(save_path / 'Ebay_train.txt', sep=' ')
    df_test  = pd.read_csv(save_path / 'Ebay_test.txt', sep=' ')
    
    df_info['label'] = df_info['class_id'].apply(lambda x: f'sop_{x}')
    df_info['is_test'] = df_info.path.isin(df_test.path).astype(int)
    df_info = df_info.rename(columns={'path' : 'file_name'})
    df_info = add_image_sizes(df_info, dataset_path)
    df_info = df_info[['file_name', 
                       'label', 
                       'width', 
                       'height',
                       'is_test']]
    
    df_info.to_csv(save_path / 'dataset_info.csv', index=False) 

    