from pathlib import Path
import pandas as pd
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig


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
        df_full = df_train
    else:
        if 'fold' not in df_folds:
            df_folds['fold'] = 0

        df_train = df_folds[((df_folds.fold != fold) & 
                            (df_folds.fold >= 0)) | 
                            (df_folds.fold == -1)]
        df_valid = df_folds[((df_folds.fold == fold) & 
                            (df_folds.fold >= 0)) | 
                            (df_folds.fold == -2)]
        df_full = df_folds

    return df_full, df_train, df_valid


def get_train_val_split(annotation, fold=0):
    if isinstance(annotation, list):
        df_train = []
        df_valid = []
        df_full  = []
        for annotation_file in annotation:
            (df_full_i, 
             df_train_i, 
             df_valid_i) = get_train_val_from_file(annotation_file, 
                                                   fold=fold)
            df_full.append(df_full_i)
            df_train.append(df_train_i)
            df_valid.append(df_valid_i)
            

        df_full = pd.concat(df_full, ignore_index=True, sort=False)
    else:
        (df_full, 
         df_train, 
         df_valid) = get_train_val_from_file(annotation, 
                                             fold=fold)
        
    return df_train, df_valid, df_full