from pathlib import Path
import pandas as pd
import numpy as np
import json
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
from dataclasses import dataclass


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


def get_train_val_from_file(annotation_file, fold=0):
    df_folds = read_pd(annotation_file)
    if 'is_test' in df_folds:
        df_train = df_folds[df_folds['is_test']==0]
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
             df_valid_i) = get_train_val_from_file(annotation_file, 
                                                   fold=fold)
            df_train.append(df_train_i)
            df_valid.append(df_valid_i)
            
    else:
        (df_train, 
         df_valid) = get_train_val_from_file(annotation, 
                                             fold=fold)
        
    return df_train, df_valid


@dataclass
class DatasetStats():
    n_classes_total: int
    n_classes_train: int
    n_classes_valid: int
    n_samples_total: int
    n_samples_train: int
    n_samples_valid: int
    label_counts: dict
    id_counts: dict

    def __repr__(self):
        tab_string = '    '
        repr_str = ''
        repr_str += f'{tab_string*2}classes total : {self.n_classes_total}\n'
        repr_str += f'{tab_string*2}classes train : {self.n_classes_train}\n'
        repr_str += f'{tab_string*2}classes valid : {self.n_classes_valid}\n'
        repr_str += f'{tab_string*2}samples total : {self.n_samples_total}\n'
        repr_str += f'{tab_string*2}samples train : {self.n_samples_train}\n'
        repr_str += f'{tab_string*2}samples valid : {self.n_samples_valid}\n'
        return repr_str


def get_dataset_stats(
        df_train, 
        labels_to_ids,
        df_valid=None,
        label_column='label'
    ):
    
    if isinstance(df_train, list):
        df_train = pd.concat(
            df_train, 
            ignore_index=True, 
            sort=False
        )

    if df_valid is not None:
        if isinstance(df_valid, list):
            df_valid = pd.concat(
                df_valid, 
                ignore_index=True, 
                sort=False
            )
        
        df_full = pd.concat(
            [df_train, df_valid], 
            ignore_index=True, 
            sort=False
        )
        n_classes_valid = df_valid[label_column].nunique()
        n_samples_valid = len(df_valid)
    else:
        df_full = df_train
        n_classes_valid = 0
        n_samples_valid = 0

    label_counts = dict(df_full[label_column].value_counts())
    id_counts = {labels_to_ids[k]: int(v) for k, v in label_counts.items()}

    dataset_stats = DatasetStats(
        n_classes_total=df_full[label_column].nunique(),
        n_classes_train=df_train[label_column].nunique(),
        n_classes_valid=n_classes_valid,
        n_samples_total=len(df_full),
        n_samples_train=len(df_train),
        n_samples_valid=n_samples_valid,
        label_counts=label_counts,
        id_counts=id_counts,
    )

    return dataset_stats