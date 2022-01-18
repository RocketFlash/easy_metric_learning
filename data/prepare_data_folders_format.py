import pandas as pd
from pathlib import Path
import json
from utils import get_labels_to_ids_map, get_stratified_kfold

def make_all_training(df):
    train_df = df
    train_df['fold'] = -1

    train_df['fold'] = train_df['fold'].astype(int)
    return train_df[['file_name', 'label','label_id', 'fold']]


def make_all_testing(df):
    train_df = df
    train_df['fold'] = -2

    train_df['fold'] = train_df['fold'].astype(int)
    return train_df[['file_name', 'label','label_id', 'fold']]


if __name__ == '__main__':
    DATASET_PATH = Path('/home/rauf/datasets/retechlabs/metric_learning/final_test/products_images/')
    CLASSES_FOLDER_PATHS = sorted(list(DATASET_PATH.glob('*/')))
    
    RANDOM_STATE = 28
    EXT = '.png'
    
    labels = []
    image_names = []

    for class_folder_path in CLASSES_FOLDER_PATHS:
        images = sorted(list(class_folder_path.glob(f'*{EXT}')))
        label = class_folder_path.name
        for img in images:
            labels.append(label)    
            image_names.append(f'{label}/{img.name}')

    labels_to_ids, ids_to_labels = get_labels_to_ids_map(labels)
    with open(DATASET_PATH / 'labels_to_ids.json', 'w') as fp:
        json.dump(labels_to_ids, fp)
    
    with open(DATASET_PATH / 'ids_to_labels.json', 'w') as fp:
        json.dump(ids_to_labels, fp)

    
    df = pd.DataFrame(list(zip(image_names, labels)), columns =['file_name', 'label'])
    df['label_id'] = df['label'].map(labels_to_ids)

    df = get_stratified_kfold(df, k=2, random_state=28)
    df.to_csv(DATASET_PATH / 'folds.csv', index=False)

    print(df)

    df_folds_train = make_all_training(df)
    df_folds_test = make_all_testing(df)

    df_folds_train.to_csv(DATASET_PATH / 'folds_train.csv', index=False)
    df_folds_test.to_csv(DATASET_PATH / 'folds_test.csv', index=False)