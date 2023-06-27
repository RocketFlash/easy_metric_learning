import argparse
import pandas as pd
from pathlib import Path
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
        download_dataset(save_path, dataset='cub')
        dataset_path = save_path / 'CUB_200_2011'
    else:
        dataset_path = Path(args.dataset_path)
    save_path = dataset_path

    df_images   = pd.read_csv(save_path / 'images.txt', 
                              sep=' ', 
                              names=['id', 'file_name'])
    df_bboxes   = pd.read_csv(save_path / 'bounding_boxes.txt', 
                              sep=' ', 
                              names=['id', 'x1', 'y1', 'x2', 'y2'],
                              dtype = { 'id': int,
                                        'x1': int,
                                        'y1': int,
                                        'x2': int,
                                        'y2': int})
    bboxes    = df_bboxes[['x1', 'y1', 'x2', 'y2']].values.astype(str)
    boxes_str = [' '.join(bbox) for bbox in bboxes]
    df_bboxes['bbox'] = boxes_str
    df_bboxes = df_bboxes[['id', 'bbox']]

    df_classes  = pd.read_csv(save_path / 'classes.txt', 
                              sep=' ',
                              names=['class_id', 'label'])
    df_img_to_class  = pd.read_csv(save_path / 'image_class_labels.txt', 
                                   sep=' ',
                                   names=['id', 'class_id'])
    df_img_to_class = df_img_to_class.assign(is_test=[1 if x > 100 else 0 for x in df_img_to_class['class_id']])
    
    df_img_to_class['label'] = df_img_to_class["class_id"].map(dict(df_classes.values))
    df_img_to_class = df_img_to_class[['id', 'label', 'is_test']]

    df_info = pd.merge(df_images, df_img_to_class, on="id")
    df_info = pd.merge(df_info, df_bboxes, on="id")

    df_info = add_image_sizes(df_info, 
                              dataset_path, 
                              with_images_folder=True)
    df_info = df_info[['file_name', 
                       'label', 
                       'width', 
                       'height',
                       'bbox',
                       'is_test']]
    
    df_info.to_csv(save_path / 'dataset_info.csv', index=False) 

    