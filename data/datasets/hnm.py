import zipfile
import os
import pandas as pd
from pathlib import Path
from .base_dataset import BaseDataset


class HNM(BaseDataset):
    def __init__(self, save_path):
        super(HNM, self).__init__(save_path=save_path)
        self.dataset_folder = Path(self.dataset_path) / 'hnm'


    def download(self):
        if not self.dataset_folder.is_dir():
            self.dataset_folder.mkdir(exist_ok=True)

            try:
                print(f'Downloading dataset...')
                os.system(f"kaggle competitions download -c h-and-m-personalized-fashion-recommendations -p {str(self.dataset_folder)}")

                z_file = self.dataset_folder / 'h-and-m-personalized-fashion-recommendations.zip'
                with zipfile.ZipFile(z_file, 'r') as z_f:
                    z_f.extractall(str(self.dataset_folder))

                (self.dataset_folder / 'transactions_train.csv').unlink()
                (self.dataset_folder / 'sample_submission.csv').unlink()
                z_file.unlink()
                print(f'Dataset have been downloaded and extracted')
            except:
                print('In order to download H&M dataset you must have kaggle account and generated kaggle.json file in ~/.kaggle/kaggle.json ')
        

    def prepare(self):
        df_info  = pd.read_csv(self.dataset_folder / 'articles.csv')
        product_code_unique = df_info.product_code.unique()
        mapping_table = {}
        for n, i in enumerate(product_code_unique):
            mapping_table[i] = n
        df_info.product_code = df_info.product_code.map(lambda x: mapping_table[x])
        l_f = lambda x: (self.dataset_folder / f'images/{x}').is_file()
        l_f_n = lambda x: f'0{str(x)[:2]}/0{str(x)}.jpg'
        df_info['file_name'] = df_info.article_id.map(l_f_n)
        df_info['avaliable'] = df_info.file_name.map(l_f)
        df_info.drop(df_info[df_info.avaliable == False].index, inplace = True)

        df_info = df_info.rename(columns={
            'article_id' : 'label',
            'detail_desc': 'title'
        })
        df_info = df_info[['file_name', 'label' , 'title']]
        df_info['label'] = df_info['label'].apply(lambda x: f'hnm_{x}')
        df_info.reset_index(drop=True, inplace=True)
        df_info['is_test'] = 0
        
        images_path = self.dataset_folder / 'images'
        df_info = self.add_image_sizes(df_info, images_path)

        print(df_info)
        print(df_info.dtypes)
        df_info.to_csv(self.dataset_folder / 'dataset_info.csv', index=False) 
        
        
        
        