import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm

if __name__ == '__main__':
    DATASET_PATH = Path('/home/rauf/datasets/retechlabs/metric_learning/final_test')
    IMAGES_PATH = DATASET_PATH / 'JPEGImages_test'
    ANNOTATION_FILE_PATH = IMAGES_PATH / 'xml_parsed_test.csv'
    SAVE_PATH = DATASET_PATH / 'products_images'
    SAVE_PATH.mkdir(exist_ok=True) 

    annotations = pd.read_csv(ANNOTATION_FILE_PATH)
    grouped = annotations.groupby('keyframeName')

    for name, group in tqdm(grouped):
        image_path = IMAGES_PATH / name
        image = cv2.imread(str(image_path))
        img_name = name.split('.')[0]

        for index, row in group.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            product_image = image[y1:y2, x1:x2]
            product_upc = row['upc']
            
            SAVE_PATH_UPC = SAVE_PATH / str(product_upc)
            SAVE_PATH_UPC.mkdir(exist_ok=True)
            SAVE_PATH_IMAGE = SAVE_PATH_UPC / f'{img_name}_{index}.png'

            cv2.imwrite(str(SAVE_PATH_IMAGE), product_image)

