import argparse
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import imagesize
import numpy as np
import wget
import zipfile


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


def download_dataset(save_path):
    DATASET_URL = 'ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip'
    print(f'Download dataset from : {DATASET_URL}')

    wget.download(DATASET_URL, out=str(save_path))
    z_file = save_path / 'Stanford_Online_Products.zip'
    with zipfile.ZipFile(z_file, 'r') as z_f:
        z_f.extractall(str(save_path))
    z_file.unlink()
    print(f'Dataset was downloaded and extracted')


def add_image_sizes(df, dataset_path):
    image_sizes = []

    for index, row in df.iterrows():
        image_path = dataset_path / row.file_name
        width, height = imagesize.get(image_path)
        image_sizes.append([width, height])

    image_sizes = np.array(image_sizes)
    df['width']  = list(image_sizes[:, 0])
    df['height'] = list(image_sizes[:, 1])
    return df
    

if __name__ == '__main__':
    args = parse_args()

    if args.download:
        save_path = Path(args.save_path)
        save_path.mkdir(exist_ok=True)
        download_dataset(save_path)
        dataset_path = save_path / 'Stanford_Online_Products'
    else:
        dataset_path = Path(args.dataset_path)
    save_path = dataset_path

    df_info  = pd.read_csv(save_path / 'Ebay_info.txt', sep=' ')
    df_train = pd.read_csv(save_path / 'Ebay_train.txt', sep=' ')
    df_test  = pd.read_csv(save_path / 'Ebay_test.txt', sep=' ')
    
    df_info['label'] = df_info['path'].apply(lambda x: x.split('/')[0])
    df_info['is_test'] = df_info.path.isin(df_test.path).astype(int)
    df_info = df_info.rename(columns={'path' : 'file_name'})
    df_info = add_image_sizes(df_info, dataset_path)
    df_info = df_info[['file_name', 
                       'label', 
                       'width', 
                       'height',
                       'is_test']]
    
    df_info.to_csv(save_path / 'dataset_info.csv', index=False) 

    