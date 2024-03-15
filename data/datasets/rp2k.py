import wget
import zipfile
import pandas as pd
from pathlib import Path
from .base import BaseDataset


class RP2K(BaseDataset):
    def __init__(self, save_path):
        super(RP2K, self).__init__(save_path=save_path)
        self.dataset_folder = Path(self.dataset_path) / 'rp2k'
        self.dataset_url = 'https://blob-nips2020-rp2k-dataset.obs.cn-east-3.myhuaweicloud.com/rp2k_dataset.zip'
       

    def download(self):
        if not self.dataset_folder.is_dir():
            self.dataset_folder.mkdir(exist_ok=True)

            z_file  = self.dataset_folder / 'rp2k_dataset.zip'
            wget.download(self.dataset_url, out=str(z_file))

            with zipfile.ZipFile(z_file, 'r') as z_f:
                z_f.extractall(str(self.dataset_folder))
            z_file.unlink()

            print(f'Dataset have been downloaded and extracted')


    def prepare(self):
        train_path = self.dataset_folder / 'all' / 'train'
        test_path  = self.dataset_folder / 'all' / 'test'

        df_train = self.get_labels_and_paths(train_path, split='all/train')
        df_test  = self.get_labels_and_paths(test_path,  split='all/test')

        df_train['is_test'] = 0
        df_test['is_test']  = 1

        df_info = pd.concat(
            [df_train, df_test],
            ignore_index=True
        )
        
        df_info['label'] = df_info['label'].apply(lambda x: f'rp2k_{x}')
        
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
        
        
        
        