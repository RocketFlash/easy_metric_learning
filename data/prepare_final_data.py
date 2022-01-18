import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time 
import shutil

if __name__ == '__main__':
    DATASET_PATH = Path('/home/rauf/datasets/retechlabs/metric_learning/final_test')
    IMAGES_PATH = DATASET_PATH / 'bbox'
    ANNOTATION_FILE_PATH = DATASET_PATH / 'ref_upc.txt'
    SAVE_PATH = DATASET_PATH / 'products_images_gt'
    SAVE_PATH.mkdir(exist_ok=True) 

    annotations = []
    with open(ANNOTATION_FILE_PATH) as file_f:
        for l_i, line in tqdm(enumerate(file_f), total=97878):
            product_upc = line.rstrip()
            annotations.append(product_upc)
            image_id = str(l_i).zfill(4)
            image_path = IMAGES_PATH / f'bbox_{image_id}.png'
            SAVE_PATH_UPC = SAVE_PATH / str(product_upc)
            SAVE_PATH_UPC.mkdir(exist_ok=True)
            SAVE_PATH_IMAGE = SAVE_PATH_UPC / f'bbox_{image_id}.png'
            shutil.copy(image_path, SAVE_PATH_IMAGE)

    print(len(annotations))

