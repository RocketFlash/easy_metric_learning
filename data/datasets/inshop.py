import zipfile
import gdown
import pandas as pd
from pathlib import Path
from .base import BaseDataset


class Inshop(BaseDataset):
    def __init__(self, save_path):
        super(Inshop, self).__init__(save_path=save_path)
        self.dataset_folder = Path(self.dataset_path) / 'inshop'
        self.dataset_url = 'https://drive.google.com/drive/folders/0B7EVK8r0v71pVDZFQXRsMDZCX1E?resourcekey=0-4R4v6zl4CWhHTsUGOsTstw'
        self.images_url = '0B7EVK8r0v71pS2YxRE1QTFZzekU'
        self.split_url  = '0B7EVK8r0v71pYVBqLXpRVjhHeWM'


    def download(self):
        if not self.dataset_folder.is_dir():
            # File ids from InShop dataset: https://drive.google.com/drive/folders/0B7EVK8r0v71pVDZFQXRsMDZCX1E?resourcekey=0-4R4v6zl4CWhHTsUGOsTstw
            print(f'Download dataset from : {self.dataset_url}')

            self.dataset_folder.mkdir(exist_ok=True)
            z_file_path     = self.dataset_folder / 'img.zip'
            split_file_path = self.dataset_folder / 'list_eval_partition.txt'

            gdown.download(id=self.images_url, 
                        output=str(z_file_path), 
                        quiet=False)

            with zipfile.ZipFile(z_file_path, 'r') as z_f:
                z_f.extractall(str(self.dataset_folder))

            z_file_path.unlink()

            gdown.download(id=self.split_url, 
                        output=str(split_file_path), 
                        quiet=False)
            print(f'Dataset have been downloaded and extracted')


    def prepare(self):
        df_info   = pd.read_csv(self.dataset_folder / 'list_eval_partition.txt', 
                                sep='\s+', 
                                skiprows=1)
        df_info = df_info.rename(columns={'image_name' : 'file_name',
                                        'item_id' : 'label'})
        df_info = df_info.assign(is_test=[0 if x == 'train' else 1 for x in df_info['evaluation_status']])
        df_info = self.add_image_sizes(df_info, 
                                       self.dataset_folder)

        df_info['label'] = df_info['label'].apply(lambda x: f'inshop_{x}')
        df_info = df_info[['file_name', 
                        'label', 
                        'width', 
                        'height',
                        'evaluation_status',
                        'is_test']]
        
        print(df_info)
        print(df_info.dtypes)
        
        df_info.to_csv(self.dataset_folder / 'dataset_info.csv', index=False) 