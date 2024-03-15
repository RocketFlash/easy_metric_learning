import tarfile
import wget
import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from .base import BaseDataset


class Inaturalist2021(BaseDataset):
    def __init__(self, save_path):
        super(Inaturalist2021, self).__init__(save_path=save_path)
        self.dataset_folder = Path(self.dataset_path) / 'inaturalist_2021'
        self.train_data_url = 'https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.tar.gz'
        self.valid_data_url = 'https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz'
        self.train_anno_url = 'https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.json.tar.gz'
        self.valid_anno_url = 'https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.json.tar.gz'


    def download(self):
        if not self.dataset_folder.is_dir():
            train_data_tar_file = self.dataset_folder / 'train.tar.gz'
            valid_data_tar_file = self.dataset_folder / 'val.tar.gz'
            train_anno_tar_file = self.dataset_folder / 'train.json.tar.gz'
            valid_anno_tar_file = self.dataset_folder / 'val.json.tar.gz'

            wget.download(self.train_data_url, out=str(train_data_tar_file))
            wget.download(self.valid_data_url, out=str(valid_data_tar_file))
            wget.download(self.train_anno_url, out=str(train_anno_tar_file))
            wget.download(self.valid_anno_url, out=str(valid_anno_tar_file))

            with tarfile.open(train_data_tar_file) as tar:
                tar.extractall(self.dataset_folder)
            train_data_tar_file.unlink()

            with tarfile.open(valid_data_tar_file) as tar:
                tar.extractall(self.dataset_folder)
            valid_data_tar_file.unlink()

            with tarfile.open(train_anno_tar_file) as tar:
                tar.extractall(self.dataset_folder)
            train_anno_tar_file.unlink()

            with tarfile.open(valid_anno_tar_file) as tar:
                tar.extractall(self.dataset_folder)
            valid_anno_tar_file.unlink()

            print(f'Dataset have been downloaded and extracted')


    def parse_json_annos(self, file_path):
        file_names = []
        labels  = []
        widths  = []
        heights = []

        with open(file_path) as ann_file:
            anno_dict = json.load(ann_file)
            categories_dict = {cat['id']: cat for cat in anno_dict['categories']}
            images_dict = {img['id']: img for img in anno_dict['images']}
            
            for anno in tqdm(anno_dict['annotations']):
                img_id = anno['image_id']
                cat_id = anno['category_id']

                image_info = images_dict[img_id]
                label = categories_dict[cat_id]['name'].replace(' ', '_')
                label = f'{cat_id}_{label}'
                file_names.append(image_info['file_name'])
                labels.append(label)
                widths.append(int(image_info['width']))
                heights.append(int(image_info['height']))

        return dict(
            file_name=file_names,
            label=labels,
            width=widths,
            height=heights
        )


    def prepare(self):
        data_train = self.parse_json_annos(self.dataset_folder / 'train.json')
        data_valid = self.parse_json_annos(self.dataset_folder / 'val.json')

        df_train = pd.DataFrame(data_train)
        df_valid = pd.DataFrame(data_valid)

        df_train['is_test'] = 0
        df_valid['is_test'] = 1

        df_info = pd.concat(
            [df_train, df_valid],
            ignore_index=True
        )

        print(df_info)
        print(df_info.dtypes)

        df_info.to_csv(self.dataset_folder / 'dataset_info.csv', index=False) 

        