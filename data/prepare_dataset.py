import pandas as pd
from pathlib import Path
import json
from utils import get_stratified_kfold, make_all_training, make_all_testing
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='prepare dataset for training')
    parser.add_argument('--dataset_path', default="", help="path to the dataset")
    parser.add_argument('--dataset_csv', default="", help="path to the csv file describing dataset")
    parser.add_argument('--k', default=5, type=int, help="number of folds")
    parser.add_argument('--random_seed', default=28, help='random seed')
    parser.add_argument('--save_name', default="folds", help="name of saved files")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    RANDOM_STATE = args.random_seed

    if args.dataset_csv:
        DATASET_CSV = Path(args.dataset_csv)
        DATASET_PATH = DATASET_CSV.parents[0]

        df = pd.read_csv(DATASET_CSV, dtype={
                                             'label': str,
                                             'file_name': str,
                                             'width': int,
                                             'height': int
                                            })
    else:
        DATASET_PATH = Path(args.dataset_path)
        CLASSES_FOLDER_PATHS = sorted(list(DATASET_PATH.glob('*/')))

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

        df = pd.DataFrame(list(zip(image_names, labels)), columns =['file_name', 'label'])
    
    counts = df['label'].value_counts()
    counts_df = pd.DataFrame({
                              'label':counts.index, 
                              'frequency':counts.values
                             })
    print(counts_df.describe())

    df_folds_train = make_all_training(df)
    df_folds_test = make_all_testing(df)

    df_folds_train.to_csv(DATASET_PATH / f'{args.save_name}_train_only.csv', index=False)
    df_folds_test.to_csv(DATASET_PATH / f'{args.save_name}_test_only.csv', index=False)

    if args.k>0:
        df = get_stratified_kfold(df, k=args.k, random_state=RANDOM_STATE)

    df.to_csv(DATASET_PATH / f'{args.save_name}.csv', index=False)
    print(df)

    