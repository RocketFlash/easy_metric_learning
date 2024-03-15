import wget
import zipfile
import pandas as pd
from pathlib import Path
from .base import BaseDataset


class Products10K(BaseDataset):
    def __init__(self, save_path):
        super(Products10K, self).__init__(save_path=save_path)
        self.dataset_folder = Path(self.dataset_path) / 'products10k'


    def download(self):
        if not self.dataset_folder.is_dir():
            self.dataset_folder.mkdir(exist_ok=True)

            TRAIN_IMAGES_URL = 'https://hxppaq.bl.files.1drv.com/y4mRRNNq8uUa-jR4FBBllPtxas1R00_ytt5IIXPFIWVZfxbBndfVZRRUebeWs9nWE3aowktixlQsXNZhFes-Cr_P26suWxEAA72YK1AsvNMSbqpxunzqxtGoPOanyS6xVM3lRDg0kol8HljzHnQ3rgJTmwb4qEX5g_TBoCvgE2bX7RdX-zWt1JnIDeqQrJDiMEayBMagPrKI7ld-flEqenCIg'
            TEST_IMAGES_URL  = 'https://hxppaq.bl.files.1drv.com/y4mM4VFu53lo1i8OW7HhQlmP5YJANItp3B0Wc8UAD4V84pPmy5arhJdpxpvS-mpk_6Rv9POdJnpqpNnOqJ39DR3FfG5rhMisAztLk-wi7ZCQ0F63N1gZRVkz6NQMZLNamTfo818P6tWficovSKTFASeWmdh_q-lp6Pkly6kPo5KREvqwXaFKZAb40duubnevFntFeIqNx78HhwwDJVWgS-r-A'
            TRAIN_ANNO_URL   = 'https://hxppaq.dm.files.1drv.com/y4mpxYK-sFQ95BwB6-uRsrREZ2lr7tSxfp2gkHisKljpflkRP-lgQHBNX91Bh7Q-1_GKqJ5NMJ3_97AixtUvW875pK4OLj2js2Ga5jiFabQfycYTzG8MaJfWHHFcoA6cK0vrn6M_sqGFqobl4zNFCOXHQ'
            TEST_ANNO_URL    = 'https://hxppaq.dm.files.1drv.com/y4mdENOMPzuyOBGAwht99-BJjAen3ZPzRPJHz5cvxH0qK635N2Nh7E8tE8ZLFO2bXNpaPkXLGRW7RRWLdF0nSXR_OUnLyRb9s2JOykBzCklduYht_uUoN9vL9ZgKJKV-tnt4XKytFYtOJXbObNfSN8C5A' 
            save_train_zip_file  = self.dataset_folder / 'train.zip'
            save_test_zip_file   = self.dataset_folder / 'test.zip'
            save_train_anno_file = self.dataset_folder / 'train.csv'
            save_test_anno_file  = self.dataset_folder / 'test_kaggletest.csv'
            wget.download(TRAIN_IMAGES_URL, out=str(save_train_zip_file))
            wget.download(TEST_IMAGES_URL, out=str(save_test_zip_file))
            wget.download(TRAIN_ANNO_URL, out=str(save_train_anno_file))
            wget.download(TEST_ANNO_URL, out=str(save_test_anno_file))

            with zipfile.ZipFile(save_train_zip_file, 'r') as z_f:
                z_f.extractall(str(self.dataset_folder))
            with zipfile.ZipFile(save_test_zip_file, 'r') as z_f:
                z_f.extractall(str(self.dataset_folder))
            save_train_zip_file.unlink()
            save_test_zip_file.unlink()

            print(f'Dataset have been downloaded and extracted')


    def prepare(self):
        df_train  = pd.read_csv(self.dataset_folder / 'train.csv')
        df_train = df_train[['name', 'class']]
        df_train['is_test'] = 0
        df_train['name'] = df_train['name'].apply(lambda x: f'train/{x}')

        df_test  = pd.read_csv(self.dataset_folder / 'test_kaggletest.csv')
        df_test = df_test[['name', 'class']]
        df_test['is_test'] = 1
        df_test['name'] = df_test['name'].apply(lambda x: f'test/{x}')

        df_info = pd.concat(
            [df_train, df_test],
            ignore_index=True
        )
        df_info = df_info.rename(columns={'name' : 'file_name',
                                        'class' : 'label'})
        
        df_info['label'] = df_info['label'].apply(lambda x: f'prod10k_{x}')
        
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
        
        
        
        