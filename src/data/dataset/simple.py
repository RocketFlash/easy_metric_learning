from os.path import join
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from .utils import get_labels_to_ids_map


class MLDataset(Dataset):
    """Metric learning Dataset. Read images, apply 
       augmentation and preprocessing transformations.
    
    Args:
        root_dir (str): path to data folder
        df_annos (str): dataframe with names
        transform (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
    
    """
    
    def __init__(
            self,
            root_dir,
            df_annos,   
            transform=None,
            label_column='label',
            fname_column='file_name',
            return_filenames=False,
            labels_to_ids=None,
            use_bboxes=False
        ):

        self.images_paths = []
        self.labels = []
        self.bboxes = []
        self.return_filenames = return_filenames
        self.use_bboxes = use_bboxes
        
        if isinstance(root_dir, list) and isinstance(df_annos, list):
            lines = [df_nms[fname_column].tolist() for df_nms in df_annos]
            labels = [df_nms[label_column].tolist() for df_nms in df_annos]
            for ln_idx, lns in enumerate(lines):
                root_dir_i = root_dir[ln_idx]
                if (Path(root_dir_i) / 'images').is_dir():
                    root_dir_i = str(Path(root_dir_i)  / 'images')
                self.images_paths += [join(root_dir_i, str(i)) for i in lns]
            for lab_idx, lbl in enumerate(labels):
                self.labels += lbl  
            if self.use_bboxes:
                if 'bbox' in  df_annos[0]:
                    bboxes = [df_nms['bbox'].tolist() for df_nms in df_annos]
                    for bbxs in bboxes:
                        bbxs = [bbox.split(' ') for bbox in bbxs]
                        self.bboxes += bbxs
        else:
            lines = df_annos[fname_column].tolist()
            imgs_pth = Path(root_dir) / 'images'
            if imgs_pth.is_dir():
                self.images_paths = [join(root_dir, 'images', str(i)) for i in lines]
            else:
                self.images_paths = [join(root_dir, str(i)) for i in lines]
            self.labels = df_annos[label_column].tolist()
            if self.use_bboxes:
                if 'bbox' in  df_annos:
                    bboxes = df_annos['bbox'].tolist()
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
        
        self.file_names = lines
        self.transform = transform
    

    def get_labels_to_ids(self):
        return self.labels_to_ids
    

    def get_ids_to_labels(self):
        return self.ids_to_labels
    

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
            
            if self.transform:
                sample = self.transform(image=image)
                image = sample['image']

            data = torch.tensor(self.label_ids[i], 
                                dtype=torch.long)
                
            if self.return_filenames:
                return image, data, self.file_names[i]
            else:
                return image, data
        except:
            print(f'Corrupted image {image_path}')
        
        return None
        
    def __len__(self):
        return len(self.images_paths)