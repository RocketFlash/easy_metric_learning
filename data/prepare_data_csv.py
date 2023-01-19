import pandas as pd
from pathlib import Path
import json
from utils import get_labels_to_ids_map, get_stratified_kfold, make_all_training, make_all_testing
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='prepare dataset for training using csv')
    # arguments from command line
    parser.add_argument('--dataset_csv', default="./", help="path to the csv file describing dataset")
    parser.add_argument('--k', default=5, type=int, help="number of folds")
    parser.add_argument('--random_seed', default=28, help='random seed')

    args = parser.parse_args()

    DATASET_CSV = Path(args.dataset_csv)
    DATASET_PATH = DATASET_CSV.parents[0]
    RANDOM_STATE = args.random_seed

    df = pd.read_csv(DATASET_CSV, dtype={'label': str,
                                        'file_name': str,
                                        'width': int,
                                        'height': int})

    counts = df['label'].value_counts()
    counts_df = pd.DataFrame({'label':counts.index, 
                              'frequency':counts.values})
    print(counts_df.describe())

    labels = df['label'].unique()

    labels_to_ids, ids_to_labels = get_labels_to_ids_map(labels)
    with open(DATASET_PATH / 'labels_to_ids.json', 'w') as fp:
        json.dump(labels_to_ids, fp)
    
    with open(DATASET_PATH / 'ids_to_labels.json', 'w') as fp:
        json.dump(ids_to_labels, fp)

    df['label_id'] = df['label'].map(labels_to_ids)

    if args.k>0:
        df_folds_train = make_all_training(df)
        df_folds_test = make_all_testing(df)

        df_folds_train.to_csv(DATASET_PATH / 'folds_train_only.csv', index=False)
        df_folds_test.to_csv(DATASET_PATH / 'folds_test_only.csv', index=False)

        df = get_stratified_kfold(df, k=args.k, random_state=RANDOM_STATE)

    df.to_csv(DATASET_PATH / 'folds.csv', index=False)
    print(df)

    