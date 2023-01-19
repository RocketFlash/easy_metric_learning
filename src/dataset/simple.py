import os
from os.path import join, split, isdir, isfile, abspath
from pathlib import Path
import numpy as np
import cv2
from imageio import mimread
from torch.utils.data import Dataset as BaseDataset
import matplotlib.pyplot as plt

class MetricDataset(BaseDataset):
    """Metric learning Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        root_dir (str): path to data folder
        df_names (str): dataframe with names
        transform (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
    
    """
    
    def __init__(
            self,
            root_dir,
            df_names,   
            transform=None,
            return_filenames=False):

        self.images_paths = []
        self.labels = []
        self.return_filenames = return_filenames
        
        if isinstance(root_dir, list) and isinstance(df_names, list):
            lines = [df_nms['file_name'].tolist() for df_nms in df_names]
            labels = [df_nms['label_id'].tolist() for df_nms in df_names]
            for ln_idx, lns in enumerate(lines):
                self.images_paths += [join(root_dir[ln_idx], str(i)) for i in lns]
            for lab_idx, lbl in enumerate(labels):
                self.labels += lbl
                
        else:
            lines = df_names['file_name'].tolist()
            imgs_pth = Path(root_dir) / 'images'
            if imgs_pth.is_dir():
                self.images_paths = [join(root_dir, 'images', str(i)) for i in lines]
            else:
                self.images_paths = [join(root_dir, str(i)) for i in lines]
            self.labels = df_names['label_id'].tolist()
        
        self.file_names = lines
        self.augmentation = transform
        
    
    def __getitem__(self, i):
        image_path = self.images_paths[i]
        try:
            # read data
            if Path(image_path).suffix == '.gif':
                image =  plt.imread(image_path)
            else:
                image = cv2.imread(image_path)
            image = image[:,:,:3]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # apply augmentations
            if self.augmentation:
                sample = self.augmentation(image=image)
                image = sample['image']
            
            if self.return_filenames:
                return image, self.labels[i], self.file_names[i]
            else:
                return image, self.labels[i]
        except:
            print(f'Corrupted image {image_path}')
        
        return None
        
    def __len__(self):
        return len(self.images_paths)