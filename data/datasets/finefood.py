import tarfile
import wget
import os
import pandas as pd
from pathlib import Path
from .base import BaseDataset


class FineFood(BaseDataset):
    def __init__(self, save_path):
        super(FineFood, self).__init__(save_path=save_path)
        self.dataset_folder = Path(self.dataset_path) / 'largefinefoodai'


    def download(self):
        if not self.dataset_folder.is_dir():
            self.dataset_folder.mkdir(exist_ok=True)

            TRAIN_URL = 'https://s3plus.meituan.net/v1/mss_fad1a48f61e8451b8172ba5abfdbbee5/foodai-workshop-challenge/Train.tar'
            VAL_URL   = 'https://s3plus.meituan.net/v1/mss_fad1a48f61e8451b8172ba5abfdbbee5/foodai-workshop-challenge/Val.tar'

            train_file  = self.dataset_folder / 'Train.tar'
            val_file    = self.dataset_folder / 'Val.tar'

            wget.download(TRAIN_URL, out=str(train_file))
            wget.download(VAL_URL, out=str(val_file))

            with tarfile.open(train_file) as tar:
                tar.extractall(self.dataset_folder)
            train_file.unlink()

            with tarfile.open(val_file) as tar:
                tar.extractall(self.dataset_folder)
            val_file.unlink()
            print(f'Dataset have been downloaded and extracted')
        

    def prepare(self):
        train_path = self.dataset_folder / 'Train'
        val_path   = self.dataset_folder / 'Val'

        df_train = self.get_labels_and_paths(train_path, split='Train')
        df_val   = self.get_labels_and_paths(val_path, split='Val')

        df_train['is_test'] = 0
        df_val['is_test']   = 1

        df_info = pd.concat(
            [df_train, df_val],
            ignore_index=True
        )

        df_info['label'] = df_info['label'].apply(lambda x: f'finefood_{x}')

        df_info = self.add_image_sizes(df_info, self.dataset_folder)
        df_info = df_info[[
            'file_name', 
            'label', 
            'width', 
            'height',
            'is_test'
        ]]
        
        print(df_info)
        print(df_info.dtypes)
        df_info.to_csv(self.dataset_folder / 'dataset_info.csv', index=False) 
        
        
        
        