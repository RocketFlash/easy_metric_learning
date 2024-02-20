import zipfile
import wget
import pandas as pd
from pathlib import Path
from .base_dataset import BaseDataset


class SOP(BaseDataset):
    def __init__(self, save_path):
        super(SOP, self).__init__(save_path=save_path)
        self.dataset_folder = Path(self.dataset_path) / 'Stanford_Online_Products'
        self.dataset_url = 'http://ftp.cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip'


    def download(self):
        if not self.dataset_folder.is_dir():
            print(f'Download dataset from : {self.dataset_url}')
            wget.download(self.dataset_url, out=str(self.dataset_path))
            z_file = self.dataset_path / 'Stanford_Online_Products.zip'
            with zipfile.ZipFile(z_file, 'r') as z_f:
                z_f.extractall(str(self.dataset_path))
            z_file.unlink()
            print(f'Dataset have been downloaded and extracted')


    def prepare(self):
        df_info  = pd.read_csv(self.dataset_folder / 'Ebay_info.txt' , sep=' ')
        df_train = pd.read_csv(self.dataset_folder / 'Ebay_train.txt', sep=' ')
        df_test  = pd.read_csv(self.dataset_folder / 'Ebay_test.txt' , sep=' ')
        
        df_info['label'] = df_info['class_id'].apply(lambda x: f'sop_{x}')
        df_info['is_test'] = df_info.path.isin(df_test.path).astype(int)
        df_info = df_info.rename(columns={'path' : 'file_name'})
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