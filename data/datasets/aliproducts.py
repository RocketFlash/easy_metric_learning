import wget
import tarfile
import pandas as pd
from pathlib import Path
from .base_dataset import BaseDataset


class Aliproducts(BaseDataset):
    def __init__(self, save_path):
        super(Aliproducts, self).__init__(save_path=save_path)
        self.dataset_folder = Path(self.dataset_path) / 'aliproducts'
        self.dataset_urls = [
            'https://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/231780/AliProducts/100001585554035/train_val.part1.tar.gz',
            'https://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/231780/AliProducts/200001585540031/train_val.part2.tar.gz',
            'https://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/231780/AliProducts/300001585559032/train_val.part3.tar.gz',
            'https://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/231780/AliProducts/400001585578035/train_val.part4.tar.gz',
            'https://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/231780/AliProducts/500001585599038/train_val.part5.tar.gz',
            'https://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/231780/AliProducts/600001585536030/train_val.part6.tar.gz',
            'https://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/231780/AliProducts/700001585524033/train_val.part7.tar.gz',
            'https://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/231780/AliProducts/800001585502035/train_val.part8.tar.gz',
            'https://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/231780/AliProducts/900001585552031/train_val.part9.tar.gz'
        ]
       

    def download(self):
        if not self.dataset_folder.is_dir():
            self.dataset_folder.mkdir(exist_ok=True)
            for idx, dataset_url in enumerate(self.dataset_urls):
                save_part_file  = self.dataset_folder / f'train_val.part{idx+1}.tar.gz'
                wget.download(dataset_url, out=str(save_part_file))
                with tarfile.open(save_part_file) as tar:
                    tar.extractall(self.dataset_folder)
                save_part_file.unlink()
            print(f'Dataset have been downloaded and extracted')


    def prepare(self):
        train_path = self.dataset_folder / 'train'
        val_path   = self.dataset_folder / 'val'

        df_train = self.get_labels_and_paths(train_path, split='train')
        df_val   = self.get_labels_and_paths(val_path, split='val')

        df_train['is_test'] = 0
        df_val['is_test']   = 1

        df_info = pd.concat([df_train, df_val],
                            ignore_index=True)
        
        df_info['label'] = df_info['label'].apply(lambda x: f'ali_{x}')
        
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
        
        
        
        