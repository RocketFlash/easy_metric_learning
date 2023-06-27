import argparse
import deeplake
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import imagesize
import numpy as np


def get_images(df,
               dataset_path, 
               save_path=''):

    for image_id, row in tqdm(df.iterrows(), total=len(df)):
        file_path = dataset_path / row['file_name']
        image_name = file_path.name
        image = cv2.imread(str(file_path))

        image_path = save_path / image_name
        image = cv2.cvtColor(image, 
                             cv2.COLOR_BGR2RGB)
        x1, y1, w, h = [int(x) for x in row['bbox'].split(' ')]
        cv2.rectangle(image, (x1, y1), (x1+w, y1+h), (255, 0, 0))
        cv2.imwrite(str(image_path), image)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', 
                        type=str, 
                        default='', 
                        help='save path')
    parser.add_argument('--dataset_csv', 
                        type=str, 
                        default='', 
                        help='save path')
    parser.add_argument('--save_path', 
                        type=str, 
                        default='./results', 
                        help='save path')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    save_path = Path(args.save_path)
    save_path.mkdir(exist_ok=True)
    dataset_path = Path(args.dataset_path)

    df_dtype = {'label': str,
                'file_name': str,
                'width': int,
                'height': int,
                'is_test':int}  

    df = pd.read_csv(args.dataset_csv, 
                     dtype=df_dtype)
    
    get_images(df,
               dataset_path, 
               save_path)

    

    