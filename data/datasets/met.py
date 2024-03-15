import wget
import tarfile
import json
import pandas as pd
from pathlib import Path
from .base import BaseDataset


class MET(BaseDataset):
    def __init__(self, save_path):
        super(MET, self).__init__(save_path=save_path)
        self.dataset_folder = Path(self.dataset_path) / 'met'


    def download(self):
        if not self.dataset_folder.is_dir():
            IMAGES_URL = 'http://ptak.felk.cvut.cz/met/dataset/MET.tar.gz'
            
            self.dataset_folder.mkdir(exist_ok=True)

            print(f'Download images from : {IMAGES_URL}')
            save_tar_file = self.dataset_folder / 'MET.tar.gz'
            wget.download(IMAGES_URL, out=str(save_tar_file))

            with tarfile.open(save_tar_file) as tar:
                tar.extractall(self.dataset_folder)

            save_tar_file.unlink()

            ANNOTATIONS_URL = 'http://ptak.felk.cvut.cz/met/dataset/ground_truth.tar.gz'
            print(f'Download annotations from : {ANNOTATIONS_URL}')
            save_tar_file = self.dataset_folder / 'ground_truth.tar.gz'
            wget.download(ANNOTATIONS_URL, out=str(save_tar_file))

            with tarfile.open(save_tar_file) as tar:
                tar.extractall(self.dataset_folder)

            save_tar_file.unlink()

            TEST_IMAGES_URL = 'http://ptak.felk.cvut.cz/met/dataset/test_met.tar.gz'
            print(f'Download test images from : {TEST_IMAGES_URL}')
            save_tar_file = self.dataset_folder / 'test_met.tar.gz'
            wget.download(TEST_IMAGES_URL, out=str(save_tar_file))

            with tarfile.open(save_tar_file) as tar:
                tar.extractall(self.dataset_folder)

            save_tar_file.unlink()

            print(f'Dataset have been downloaded and extracted')
        

    def prepare(self):
        with open(self.dataset_folder / 'testset.json') as f:
            test_anno = json.load(f)
        with open(self.dataset_folder / 'valset.json') as f:
            val_anno  = json.load(f)

        test_labels = [[anno['path'], anno['MET_id']] for anno in test_anno if 'MET_id' in anno]
        val_labels  = [[anno['path'], anno['MET_id']] for anno in val_anno if 'MET_id' in anno]

        df_test = pd.DataFrame(test_labels, 
                            columns =['file_name', 'label'])
        df_val  = pd.DataFrame(val_labels, 
                            columns =['file_name', 'label'])
        df_test['is_test'] = 1
        df_val['is_test']  = 0

        df_test = pd.concat(
            [df_val, df_test],
            ignore_index=True
        )
        class_folders_path = sorted(list(self.dataset_folder.glob('MET/*/')))

        labels = []
        image_names = []

        for class_folder_path in class_folders_path:
            images = sorted([l for l in list(class_folder_path.glob('*.jpeg')) + \
                                        list(class_folder_path.glob('*.jpg')) + \
                                        list(class_folder_path.glob('*.png'))])
            label = class_folder_path.name
            for img in images:
                labels.append(label)    
                image_names.append(f'{label}/{img.name}')

        df_info = pd.DataFrame(list(zip(image_names, labels)), 
                            columns =['file_name', 'label'])
        df_info['file_name'] = df_info['file_name'].apply(lambda x: f'MET/{x}')
        df_info['is_test'] = 0

        
        df_info = pd.concat([df_info, df_test],
                            ignore_index=True)

        df_info['label'] = df_info['label'].apply(lambda x: f'met_{x}')
        
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
        
        
        
        