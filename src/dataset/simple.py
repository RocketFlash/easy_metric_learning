import os
from os.path import join, split, isdir, isfile, abspath
from pathlib import Path
import numpy as np
import cv2
from imageio import mimread
import torch
from torch.utils.data import Dataset as BaseDataset
import matplotlib.pyplot as plt
from .utils import get_labels_to_ids_map
import torch.nn.functional as F


class MetricDataset(BaseDataset):
    """Metric learning Dataset. Read images, apply 
       augmentation and preprocessing transformations.
    
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
            label_column='label',
            fname_column='file_name',
            return_filenames=False,
            labels_to_ids=None,
            use_categories=False,
            use_text_embeddings=False,
            category_column='category',
            text_embeddings_column='text_embeddings',
            categories_to_ids=None,
            use_bboxes=False):

        self.images_paths = []
        self.labels = []
        self.categories = []
        self.text_embeddings = []
        self.bboxes = []
        self.return_filenames = return_filenames
        self.use_categories = use_categories
        self.use_text_embeddings = use_text_embeddings
        self.use_bboxes = use_bboxes
        
        if isinstance(root_dir, list) and isinstance(df_names, list):
            lines = [df_nms[fname_column].tolist() for df_nms in df_names]
            labels = [df_nms[label_column].tolist() for df_nms in df_names]
            for ln_idx, lns in enumerate(lines):
                root_dir_i = root_dir[ln_idx]
                if (Path(root_dir_i) / 'images').is_dir():
                    root_dir_i = str(Path(root_dir_i)  / 'images')
                self.images_paths += [join(root_dir_i, str(i)) for i in lns]
            for lab_idx, lbl in enumerate(labels):
                self.labels += lbl  
            if use_categories:
                categories = [df_nms[category_column].tolist() for df_nms in df_names]
                for cat in categories:
                    self.categories += cat
            if use_text_embeddings:
                text_embeddings = [df_nms[text_embeddings_column].tolist() for df_nms in df_names]
                for text_emb in text_embeddings:
                    self.text_embeddings += text_emb
            if self.use_bboxes:
                if 'bbox' in  df_names[0]:
                    bboxes = [df_nms['bbox'].tolist() for df_nms in df_names]
                    for bbxs in bboxes:
                        bbxs = [bbox.split(' ') for bbox in bbxs]
                        self.bboxes += bbxs
        else:
            lines = df_names[fname_column].tolist()
            imgs_pth = Path(root_dir) / 'images'
            if imgs_pth.is_dir():
                self.images_paths = [join(root_dir, 'images', str(i)) for i in lines]
            else:
                self.images_paths = [join(root_dir, str(i)) for i in lines]
            self.labels = df_names[label_column].tolist()
            if use_categories:
                self.categories = df_names[label_column].tolist()
            if self.use_bboxes:
                if 'bbox' in  df_names:
                    bboxes = df_names['bbox'].tolist()
                    self.bboxes = [bbox.split(' ') for bbox in bboxes]

        self.labels = np.array(self.labels, dtype=str)
        if labels_to_ids is None:
            labels_names = sorted(np.unique(self.labels).tolist())
            self.labels_to_ids, self.ids_to_labels = get_labels_to_ids_map(labels_names)
        else:
            self.labels_to_ids = labels_to_ids
            self.ids_to_labels = {v:k for k,v in self.labels_to_ids.items()}
        self.label_ids = np.array([self.labels_to_ids[l] for l in self.labels])

        if self.use_bboxes:
            self.bboxes = [[int(b) for b in bbox] for bbox in self.bboxes]

        if use_categories:
            self.categories = [str(cat).split('##') for cat in self.categories]
            if categories_to_ids is None:
                categories_names = []
                for cat in self.categories:
                    categories_names += cat
                categories_names = sorted(list(set(categories_names)))
                self.categories_to_ids, self.ids_to_categories = get_labels_to_ids_map(categories_names)
            else:
                self.categories_to_ids = categories_to_ids
                self.ids_to_categories = {v:k for k,v in self.categories_to_ids.items()}

            self.category_ids = []
            for cat in self.categories:
                curr_cat = []
                for cat_i in cat:
                    curr_cat.append(self.categories_to_ids[cat_i])
                curr_cat = np.array(curr_cat)
                self.category_ids.append(curr_cat)
            self.n_categories = len(self.categories_to_ids)
        
        self.file_names = lines
        self.augmentation = transform
    

    def get_labels_to_ids(self):
        return self.labels_to_ids
    

    def get_ids_to_labels(self):
        return self.ids_to_labels
    

    def get_categories_to_ids(self):
        return self.categories_to_ids
    

    def get_ids_to_categories(self):
        return self.ids_to_categories
    

    def __getitem__(self, i):
        image_path = self.images_paths[i]
        try:
            if Path(image_path).suffix == '.gif':
                image =  plt.imread(image_path)
            else:
                image = cv2.imread(image_path)

            image = image[:,:,:3]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.bboxes:
                bbox = self.bboxes[i]
                x1, y1, w, h = bbox
                image = image[y1:(y1+h), x1:(x1+w), :]
            
            if self.augmentation:
                sample = self.augmentation(image=image)
                image = sample['image']

            data = torch.tensor(self.label_ids[i], 
                                dtype=torch.long)

            if self.use_categories:
                cat_id = F.one_hot(torch.tensor(self.category_ids[i],
                                                dtype=torch.long), 
                                    num_classes=self.n_categories).sum(dim=0).float()
                data = (data, cat_id)

            if self.use_text_embeddings:
                text_emb = torch.tensor(self.text_embeddings[i],
                                        dtype=torch.float32)
                data = (data, text_emb)
                
            if self.return_filenames:
                return image, data, self.file_names[i]
            else:
                return image, data
        except:
            print(f'Corrupted image {image_path}')
        
        return None
        
    def __len__(self):
        return len(self.images_paths)