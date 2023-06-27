import argparse
import deeplake
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import imagesize
import numpy as np


def get_images(images, 
               split='train',
               dataset_path='',
               download=False):

    dataset_path_split = dataset_path / split
    dataset_path_split.mkdir(exist_ok=True)

    images_bar = tqdm(images)
    images_bar.set_description(split)
    image_names = []
    image_sizes = []

    for image_id, image in enumerate(images_bar):
        image_name = f'{image_id}.jpg'
        image_names.append(f'{split}/{image_name}')
        
        image_path = dataset_path_split / image_name
        if download:
            image = cv2.cvtColor(image.numpy(), 
                                 cv2.COLOR_BGR2RGB)
            cv2.imwrite(str(image_path), image)
            
        width, height = imagesize.get(image_path)
        image_sizes.append([width, height])
    
    return image_names, image_sizes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', 
                        type=str, 
                        default='./', 
                        help='dataset path')
    parser.add_argument('--download', 
                        action='store_true', 
                        help='Dowload images')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    dataset_path.mkdir(exist_ok=True)

    ds_train = deeplake.load("hub://activeloop/stanford-cars-train")
    ds_test  = deeplake.load("hub://activeloop/stanford-cars-test")

    images_train = ds_train.images
    images_test  = ds_test.images

    labels_train = list(ds_train.car_models.numpy().squeeze(1))
    labels_test  = list(ds_test.car_models.numpy().squeeze(1))

    bboxes_train = list(ds_train.boxes.numpy().astype(int).squeeze(1))
    bboxes_test  = list(ds_test.boxes.numpy().astype(int).squeeze(1))

    bboxes_train = [' '.join(list(bbox.astype(str))) for bbox in bboxes_train]
    bboxes_test  = [' '.join(list(bbox.astype(str))) for bbox in bboxes_test]

    img_paths_train, img_sizes_train = get_images(images_train, 
                                                  split='train',
                                                  dataset_path=dataset_path,
                                                  download=args.download)
    
    img_paths_test,  img_sizes_test  = get_images(images_test, 
                                                  split='test',
                                                  dataset_path=dataset_path,
                                                  download=args.download)
    
    img_paths = img_paths_train + img_paths_test
    img_sizes = img_sizes_train + img_sizes_test
    img_sizes = np.array(img_sizes)
    img_widths  = list(img_sizes[:, 0])
    img_heights = list(img_sizes[:, 1])
    labels = labels_train + labels_test
    bboxes = bboxes_train +  bboxes_test

    n_train_classes = 98
    is_test = [0 if l<98 else 1 for l in labels]

    df = pd.DataFrame(list(zip(img_paths, 
                               labels, 
                               img_widths,
                               img_heights,
                               bboxes,
                               is_test)),
                      columns =['file_name', 
                                'label',
                                'width', 
                                'height',
                                'bbox',
                                'is_test'])
    
    df.to_csv(dataset_path / 'dataset_info.csv', index=False) 

    