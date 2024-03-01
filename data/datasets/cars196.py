import zipfile
import wget
import pandas as pd
from scipy.io import loadmat
from pathlib import Path
from .base_dataset import BaseDataset
import cv2


class Cars196(BaseDataset):
    def __init__(self, save_path):
        super(Cars196, self).__init__(save_path=save_path)
        self.dataset_folder = Path(self.dataset_path) / 'cars196'
        self.dataset_url = 'http://ftp.cs.stanford.edu/cs/cvgl/CARS196.zip'


    def download(self):
        if not self.dataset_folder.is_dir():
            print(f'Download dataset from : {self.dataset_url}')
            wget.download(self.dataset_url, out=str(self.dataset_path))
            z_file = self.dataset_path / 'CARS196.zip'
            with zipfile.ZipFile(z_file, 'r') as z_f:
                z_f.extractall(str(self.dataset_path))
            z_file.unlink()
            print(f'Dataset have been downloaded and extracted')


    def prepare(self):
        annotations = loadmat(self.dataset_folder / 'cars_annos.mat')['annotations'][0]

        file_names = []
        class_ids = []
        is_tests = []
        bboxes = []

        for annotation in annotations:
            file_name = annotation[0][0]
            x1 = annotation[1][0][0]
            y1 = annotation[2][0][0]
            x2 = annotation[3][0][0]
            y2 = annotation[4][0][0]
            class_id = annotation[5][0][0]
            is_test = annotation[6][0][0]

            file_names.append(file_name)
            bboxes.append(f'{x1} {y1} {x2} {y2}')
            class_ids.append(class_id)
            is_tests.append(is_test)

        df_info = pd.DataFrame(list(zip(file_names, 
                                        class_ids, 
                                        bboxes,
                                        is_tests)), 
                               columns=[
                                   'file_name', 
                                   'class_id', 
                                   'bbox',
                                   'is_test'
                                ])
        
        df_info['label'] = df_info['class_id'].apply(lambda x: f'cars_{x}')    
        df_info = self.add_image_sizes(df_info, self.dataset_folder)

        df_info = df_info[[
            'file_name', 
            'label',
            'bbox', 
            'width', 
            'height',
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