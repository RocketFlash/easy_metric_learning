import argparse
import deeplake
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import imagesize
import numpy as np


def get_images(images, 
               bboxes,
               save_path=''):

    images_bar = tqdm(images)

    for image_id, (image, bbox) in enumerate(zip(images_bar, bboxes)):
        image_name = f'{image_id}.jpg'
        
        image_path = save_path / image_name
        image = cv2.cvtColor(image.numpy(), 
                                cv2.COLOR_BGR2RGB)
        x1, y1, w, h = bbox
        cv2.rectangle(image, (x1, y1), (x1+w, y1+h), (255, 0, 0))
        cv2.imwrite(str(image_path), image)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', 
                        type=str, 
                        default='./results', 
                        help='save path')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    save_path = Path(args.save_path)
    save_path.mkdir(exist_ok=True)

    ds_train = deeplake.load("hub://activeloop/stanford-cars-train")
    # ds_test  = deeplake.load("hub://activeloop/stanford-cars-test")

    images_train = ds_train.images
    # images_test  = ds_test.images

    bboxes_train = list(ds_train.boxes.numpy().astype(int).squeeze(1))
    # bboxes_test  = list(ds_test.boxes.numpy().astype(int).squeeze(1))


    img_paths_train, img_sizes_train = get_images(images_train, 
                                                  bboxes_train,
                                                  save_path=save_path)
    
    # img_paths_test,  img_sizes_test  = get_images(images_test, 
    #                                               bboxes_test,
    #                                               save_path=save_path)

    