import argparse
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np
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
        download_dataset(save_path, dataset='products10k')
        dataset_path = save_path / 'products10k'
    else:
        dataset_path = Path(args.dataset_path)
    save_path = dataset_path

    df_train  = pd.read_csv(save_path / 'train.csv')
    df_train = df_train[['name', 'class']]
    df_train['is_test'] = 0
    df_train['name'] = df_train['name'].apply(lambda x: f'train/{x}')

    df_test  = pd.read_csv(save_path / 'test_kaggletest.csv')
    df_test = df_test[['name', 'class']]
    df_test['is_test'] = 1
    df_test['name'] = df_test['name'].apply(lambda x: f'test/{x}')

    df_info = pd.concat([df_train, df_test],
                        ignore_index=True)
    df_info = df_info.rename(columns={'name' : 'file_name',
                                      'class' : 'label'})
    
    df_info['label'] = df_info['label'].apply(lambda x: f'prod10k_{x}')
    
    df_info = add_image_sizes(df_info, dataset_path)
    df_info = df_info[['file_name', 
                       'label', 
                       'width', 
                       'height',
                       'is_test']]
    print(df_info)
    df_info.to_csv(save_path / 'dataset_info.csv', index=False) 

    