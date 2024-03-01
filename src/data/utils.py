from pathlib import Path
import pandas as pd
import numpy as np
import json
import torch
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
from dataclasses import dataclass


def get_labels_to_ids_map(labels):
    labels_to_ids = {}
    ids_to_labels = {}
    idx = 0

    for label in labels:
        if label not in labels_to_ids:
            labels_to_ids[label] = idx
            ids_to_labels[idx] = label
            idx+=1
    
    return labels_to_ids, ids_to_labels


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def get_labels_to_ids(labels_to_ids_path):
    if labels_to_ids_path is not None:
        with open(labels_to_ids_path) as file:
            labels_to_ids = json.load(file)
    else:
        labels_to_ids = None
    
    return labels_to_ids


def save_labels_to_ids(labels_to_ids, save_dir='./'):
    save_dir = Path(save_dir)
    with open(save_dir / 'labels_to_ids.json', 'w') as fp:
        json.dump(labels_to_ids, fp)


def get_object_from_omegaconf(obj):
    if isinstance(obj, ListConfig):
        return OmegaConf.to_object(obj)
    else:
        return obj


def read_pd(file_path):
    file_path = Path(file_path)

    if file_path.suffix=='.feather':
        df = pd.read_feather(file_path)
    else:
        df = pd.read_csv(
            file_path, 
            dtype={ 
                'label': str,
                'file_name': str,
                'width': int,
                'height': int,
                'hash' : str,
                'is_test':int,
            }
        )

    return df


def get_train_val_from_file(annotation_file, fold=0):
    df_folds = read_pd(annotation_file)
    if 'is_test' in df_folds:
        df_train = df_folds[df_folds['is_test']==0]
        df_valid = None
    else:
        if 'fold' not in df_folds:
            df_folds['fold'] = 0

        df_train = df_folds[((df_folds.fold != fold) & 
                            (df_folds.fold >= 0)) | 
                            (df_folds.fold == -1)]
        df_valid = df_folds[((df_folds.fold == fold) & 
                            (df_folds.fold >= 0)) | 
                            (df_folds.fold == -2)]

    return df_train, df_valid


def get_train_val_split(annotation, fold=0):
    if isinstance(annotation, list):
        df_train = []
        df_valid = []
        for annotation_file in annotation:
            (df_train_i, 
             df_valid_i) = get_train_val_from_file(
                 annotation_file, 
                 fold=fold
            )
            df_train.append(df_train_i)
            if df_valid_i is not None:
                df_valid.append(df_valid_i)
        if not df_valid:
            df_valid = None
            
    else:
        (df_train, 
         df_valid) = get_train_val_from_file(
             annotation, 
             fold=fold
        )
        
    return df_train, df_valid


def get_test_from_file(annotation_file, fold=0):
    df_folds = read_pd(annotation_file)
    if 'is_test' in df_folds:
        df_test = df_folds[df_folds['is_test']==1]
    else:
        if 'fold' not in df_folds:
            df_folds['fold'] = 0

        df_test = df_folds[((df_folds.fold == fold) & 
                            (df_folds.fold >= 0)) | 
                            (df_folds.fold == -2)]

    return df_test


def get_test_split(annotation, fold=0):
    if isinstance(annotation, list):
        df_test = []
        for annotation_file in annotation:
            df_test_i = get_test_from_file(
                 annotation_file, 
                 fold=fold
            )
            df_test.append(df_test_i)    
    else:
        df_test = get_test_from_file(
             annotation, 
             fold=fold
        )
        
    return df_test


@dataclass
class DatasetStats():
    n_classes: int
    n_samples: int
    label_counts: dict
    id_counts: dict
    split: str

    def __repr__(self):
        tab_string = '    '
        repr_str = ''
        repr_str += f'{tab_string*2}classes : {self.n_classes}\n'
        repr_str += f'{tab_string*2}samples : {self.n_samples}\n'
        return repr_str
    

def get_dataset_stats(
        df, 
        labels_to_ids,
        label_column='label',
        split='train'
    ):
    
    if isinstance(df, list):
        df = pd.concat(
            df, 
            ignore_index=True, 
            sort=False
        )

    label_counts = dict(df[label_column].value_counts())
    id_counts = {labels_to_ids[k]: int(v) for k, v in label_counts.items()}

    dataset_stats = DatasetStats(
        n_classes=df[label_column].nunique(),
        n_samples=len(df),
        label_counts=label_counts,
        id_counts=id_counts,
        split=split
    )

    return dataset_stats


def combine_dfs(df1, df2):
    if isinstance(df1, list):
        df1 = pd.concat(
            df1, 
            ignore_index=True, 
            sort=False
        )

    if isinstance(df2, list):
        df2 = pd.concat(
            df2, 
            ignore_index=True, 
            sort=False
        )
        
    return pd.concat(
        [df1, df2], 
        ignore_index=True, 
        sort=False
    )