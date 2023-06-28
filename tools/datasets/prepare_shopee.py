import sys
sys.path.append("./")

import argparse
import pandas as pd
from pathlib import Path
from utils import add_image_sizes, download_dataset
from data.utils import get_stratified_kfold


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
    parser.add_argument('--random_seed', 
                        default=28, 
                        help='random seed')
    return parser.parse_args()
    

if __name__ == '__main__':
    args = parse_args()

    if args.download:
        '''
        In order to download Shopee dataset you must have kaggle account 
        and generated kaggle.json file in ~/.kaggle/kaggle.json 
        '''
        
        save_path = Path(args.save_path)
        save_path.mkdir(exist_ok=True)
        download_dataset(save_path, dataset='shopee')
        dataset_path = save_path / 'shopee'
    else:
        dataset_path = Path(args.dataset_path)
    save_path = dataset_path

    df_info   = pd.read_csv(save_path / 'train.csv')
    df_info = df_info.rename(columns={'image' : 'file_name',
                                      'label_group' : 'label',
                                      'image_phash' : 'hash'})
    
    df_info['file_name'] = df_info['file_name'].apply(lambda x: f'train_images/{x}')
    df_info = get_stratified_kfold(df_info, 
                                   k=5, 
                                   random_state=args.random_seed)
    df_info = df_info.assign(is_test=[1 if x == 0 else 0 for x in df_info['fold']])

    df_info = add_image_sizes(df_info, 
                              dataset_path)
    
    df_info['label'] = df_info['label'].apply(lambda x: f'shopee_{x}')

    df_info = df_info[['file_name', 
                       'label',
                       'title', 
                       'width', 
                       'height',
                       'is_test',
                       'hash']]

    df_info.to_csv(save_path / 'dataset_info.csv', index=False) 

    