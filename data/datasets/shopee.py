import zipfile
import os
import pandas as pd
from pathlib import Path
from .base_dataset import BaseDataset


class Shopee(BaseDataset):
    def __init__(self, save_path):
        super(Shopee, self).__init__(save_path=save_path)
        self.dataset_folder = Path(self.dataset_path) / 'shopee'
    

    def download(self):
        if not self.dataset_folder.is_dir():
            # install kaggle python package first and prepare kaggle.json file and put it in ~/.kaggle/kaggle.json
            self.dataset_folder.mkdir(exist_ok=True)

            try:
                print(f'Downloading dataset...')
                os.system(f"kaggle competitions download -c shopee-product-matching -p {str(self.dataset_folder)}")

                z_file = self.dataset_folder / 'shopee-product-matching.zip'
                with zipfile.ZipFile(z_file, 'r') as z_f:
                    z_f.extractall(str(self.dataset_folder))

                z_file.unlink()
                print(f'Dataset have been downloaded and extracted')
            except:
                print('In order to download Shopee dataset you must have kaggle account and generated kaggle.json file in ~/.kaggle/kaggle.json ')


    def prepare(self):
        df_info   = pd.read_csv(self.dataset_folder / 'train.csv')
        df_info = df_info.rename(columns={
            'image' : 'file_name',
            'label_group' : 'label',
            'image_phash' : 'hash'
        })
        
        df_info['file_name'] = df_info['file_name'].apply(lambda x: f'train_images/{x}')
        df_info = get_stratified_kfold(df_info, 
                                       k=5, 
                                       random_state=28)
        df_info = df_info.assign(is_test=[1 if x == 0 else 0 for x in df_info['fold']])

        df_info = self.add_image_sizes(
            df_info, 
            self.dataset_folder
        )
        
        df_info['label'] = df_info['label'].apply(lambda x: f'shopee_{x}')

        df_info = df_info[[
            'file_name', 
            'label',
            'title', 
            'width', 
            'height',
            'is_test',
            'hash'
        ]]

        print(df_info)
        print(df_info.dtypes)

        df_info.to_csv(self.dataset_folder / 'dataset_info.csv', index=False) 
        
        
        
        