import pandas as pd
from pathlib import Path
import argparse
from utils import (get_stratified_kfold, 
                   make_all_training, 
                   make_all_testing)


def parse_args():
    parser = argparse.ArgumentParser(description='prepare dataset for training')
    parser.add_argument(
        '--dataset_info', 
        default="", 
        help="path to the csv file describing dataset"
    )
    parser.add_argument(
        '--k', 
        default=5, 
        type=int, 
        help="number of folds"
    )
    parser.add_argument(
        '--random_seed', 
        default=28, 
        help='random seed'
    )
    parser.add_argument(
        '--save_name', 
        default="folds", 
        help="name of saved files"
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    dataset_csv  = Path(args.dataset_info)
    dataset_path = dataset_csv.parents[0]

    df = pd.read_csv(
            dataset_csv, 
            dtype={
                    'label': str,
                    'file_name': str,
                    'width': int,
                    'height': int,
                    'hash': str
                }
        )

    df_folds_train = make_all_training(df)
    df_folds_test  = make_all_testing(df)

    df_folds_train.to_csv(dataset_path / f'{args.save_name}_train_only.csv', index=False)
    df_folds_test.to_csv(dataset_path / f'{args.save_name}_test_only.csv', index=False)

    df_kfold = get_stratified_kfold(
        df, 
        k=args.k, 
        random_state=args.random_seed
    )
    df_kfold.to_csv(dataset_path / f'{args.save_name}.csv', index=False)
    print(df_kfold)


    