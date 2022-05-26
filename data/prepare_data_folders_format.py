import pandas as pd
from pathlib import Path
import json
from utils import get_labels_to_ids_map, get_stratified_kfold
import argparse

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
    parser = argparse.ArgumentParser(description='find dublicates')
    # arguments from command line
    parser.add_argument('--dataset_path', default="./", help="path to the dataset")
    parser.add_argument('--k', default=5, help="number of folds")
    parser.add_argument('--random_seed', default=28, help='random seed')

    args = parser.parse_args()

    DATASET_PATH = Path(args.dataset_path)
    CLASSES_FOLDER_PATHS = sorted(list(DATASET_PATH.glob('*/')))
    
    RANDOM_STATE = args.random_seed
    
    labels = []
    image_names = []

    for class_folder_path in CLASSES_FOLDER_PATHS:
        images = sorted([l for l in list(class_folder_path.glob('*.jpeg')) + \
                           list(class_folder_path.glob('*.jpg')) + \
                           list(class_folder_path.glob('*.png'))])
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

    df = get_stratified_kfold(df, k=args.k, random_state=RANDOM_STATE)
    df.to_csv(DATASET_PATH / 'folds.csv', index=False)

    print(df)

    df_folds_train = make_all_training(df)
    df_folds_test = make_all_testing(df)

    df_folds_train.to_csv(DATASET_PATH / 'folds_train.csv', index=False)
    df_folds_test.to_csv(DATASET_PATH / 'folds_test.csv', index=False)