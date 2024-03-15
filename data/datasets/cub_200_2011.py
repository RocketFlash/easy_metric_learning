import tarfile
import wget
import pandas as pd
from pathlib import Path
from .base import BaseDataset
import shutil


class CUB(BaseDataset):
    def __init__(self, save_path):
        super(CUB, self).__init__(save_path=save_path)
        self.dataset_folder = Path(self.dataset_path) / 'CUB_200_2011'
        self.dataset_url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1'


    def download(self):
        if not self.dataset_folder.is_dir():
            print(f'Download dataset from : {self.dataset_url}')

            save_tar_file = self.dataset_path / 'CUB_200_2011.tgz'
            wget.download(self.dataset_url, out=str(save_tar_file))

            with tarfile.open(save_tar_file) as tar:
                tar.extractall(self.dataset_path)

            ds_path = self.dataset_path / 'CUB_200_2011'
            shutil.move(self.dataset_path / 'attributes.txt',
                        ds_path   / 'attributes.txt')

            save_tar_file.unlink()
            print(f'Dataset have been downloaded and extracted')


    def prepare(self):
        df_images = pd.read_csv(self.dataset_folder / 'images.txt', 
                                sep=' ', 
                                names=['id', 
                                       'file_name'])
        df_bboxes = pd.read_csv(self.dataset_folder / 'bounding_boxes.txt', 
                                sep=' ', 
                                names=['id', 'x1', 'y1', 'w', 'h'],
                                dtype = { 
                                    'id': int,
                                    'x1': int,
                                    'y1': int,
                                    'w': int,
                                    'h': int
                                })
        bboxes    = df_bboxes[['x1', 'y1', 'w', 'h']]
        bboxes['x2'] = bboxes[['x1', 'w']].sum(axis=1)
        bboxes['y2'] = bboxes[['y1', 'h']].sum(axis=1)
        bboxes = bboxes[['x1', 'y1', 'x2', 'y2']].values.astype(str)
        boxes_str = [' '.join(bbox) for bbox in bboxes]
        df_bboxes['bbox'] = boxes_str
        df_bboxes = df_bboxes[['id', 'bbox']]

        df_classes  = pd.read_csv(self.dataset_folder / 'classes.txt', 
                                  sep=' ',
                                  names=[
                                      'class_id', 
                                      'label'
                                    ])
        df_img_to_class  = pd.read_csv(self.dataset_folder / 'image_class_labels.txt', 
                                       sep=' ',
                                       names=[
                                           'id', 
                                           'class_id'
                                        ])
        df_img_to_class = df_img_to_class.assign(is_test=[1 if x > 100 else 0 for x in df_img_to_class['class_id']])
        
        df_img_to_class['label'] = df_img_to_class["class_id"].map(dict(df_classes.values))
        df_img_to_class = df_img_to_class[['id', 'label', 'is_test']]

        df_info = pd.merge(df_images, df_img_to_class, on="id")
        df_info = pd.merge(df_info, df_bboxes, on="id")
        df_info['file_name'] = df_info['file_name'].apply(lambda x: f'images/{x}') 

        df_info = self.add_image_sizes(df_info, 
                                       self.dataset_folder)
        df_info = df_info[[
            'file_name', 
            'label', 
            'width', 
            'height',
            'bbox',
            'is_test'
        ]]

        print(df_info)
        print(df_info.dtypes)

        # img_idx = 28
        # fname = str(self.dataset_folder / df_info.iloc[img_idx].file_name)
        # bbox  = [int(x) for x in df_info.iloc[img_idx].bbox.split(' ')]
        # image = cv2.imread(fname)
        # image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(255,0,0))
        # cv2.imwrite('img.jpg', image)
        
        df_info.to_csv(self.dataset_folder / 'dataset_info.csv', index=False) 