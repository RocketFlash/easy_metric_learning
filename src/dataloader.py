import os
from os.path import join, split, isdir, isfile, abspath
from pathlib import Path
import numpy as np
import cv2

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

from .transforms import get_transformations
from .utils import worker_init_fn, split_image_on_patches, get_image


class TilesDataset(BaseDataset):
    """SMetric learning Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        root_dir (str): path to data folder
        df_names (str): dataframe with names
        classes  (list): classes indexes
        transform (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
    
    """
    
    def __init__(
            self,
            image,
            window_size=20,
            step_size=10,
            is_horizontal=True,   
            transform=None):

        self.image = image
        self.tiles, self.tiler = split_image_on_patches(self.image, window_size=window_size,
                                                                    step_size=step_size,
                                                                    is_horizontal=is_horizontal)
        self.coords = self.tiler.crops 
            
        self.augmentation = transform
        
    
    def __getitem__(self, i):
        image=self.tiles[i]
        coords = self.coords[i]

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample['image']
            
        return image, coords
        
    def __len__(self):
        return len(self.tiles)



class MetricDataset(BaseDataset):
    """Metric learning Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        root_dir (str): path to data folder
        df_names (str): dataframe with names
        classes  (list): classes indexes
        transform (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
    
    """
    
    def __init__(
            self,
            root_dir,
            df_names,   
            transform=None):

        self.images_paths = []
        self.labels = []
        
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
            
        self.augmentation = transform
        
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_paths[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample['image']
            
        return image, self.labels[i]
        
    def __len__(self):
        return len(self.images_paths)


def get_loader(root_dir, 
               df_names, 
               batch_size=8, 
               img_size=512,
               num_thread=4,
               pin=False,
               test=False,
               split='train',
               transform_name='no_aug'):


    if test is False:
        transform = get_transformations(transform_name, image_size=(img_size,img_size))
    else:
        transform = get_transformations('test_aug', image_size=(img_size,img_size))
    
    dataset = MetricDataset(root_dir=root_dir,
                            df_names=df_names,      
                            transform=transform)
    
    shuffle = split=='train'
    drop_last = split=='train'
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_thread,
                             pin_memory=pin,
                             drop_last=drop_last,
                             worker_init_fn=worker_init_fn)
    return data_loader