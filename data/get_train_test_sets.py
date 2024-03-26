import imagesize
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
import argparse
from utils import get_counts_df
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(description='split dataset on training and testing parts')
    parser.add_argument(
        '--dataset_info', 
        default="./", 
        help="path to the dataset info file"
    )
    parser.add_argument(
        '--split_type', 
        type=str, 
        choices=[
            'freq',
            'minmax', 
        ],
        default='freq', 
        help="split strategy"
    )
    parser.add_argument(
        '--test_ratio', 
        type=float,
        default=0.1, 
        help="test set ratio"
    )
    parser.add_argument(
        '--min_freq', 
        type=int,
        default=10, 
        help="min number of samples in frequency bin to split bin, if less will add whole bin in training set"
    )
    parser.add_argument(
        '--max_n_samples', 
        type=int, 
        default=50, 
        help="max n of samples to select class for training"
    )
    parser.add_argument(
        '--min_n_samples', 
        type=int, 
        default=3, 
        help="min n of samples to be in training set"
    )
    parser.add_argument(
        '--random_seed', 
        type=int, 
        default=28, 
        help="random seed"
    )
    return parser.parse_args()


def split_by_frequency_bins(
        counts_df,
        test_ratio=0.1,
        min_frequency=10,
        random_seed=28
    ):
    
    n_samples_bins = counts_df.frequency.unique()
    train_classes_df = []
    test_classes_df  = []

    for bin in n_samples_bins:
        bin_df = counts_df[counts_df.frequency==bin].reset_index(drop=True)
        if len(bin_df)<min_frequency:
            train_classes_df.append(bin_df)
            continue

        train_bin_df, test_bin_df = train_test_split(
            bin_df, 
            test_size=test_ratio, 
            random_state=random_seed,
        )

        train_bin_df.reset_index(
            inplace=True,
            drop=True
        )

        test_bin_df.reset_index(
            inplace=True,
            drop=True
        )
        train_classes_df.append(train_bin_df)
        test_classes_df.append(test_bin_df)

    train_classes_df = pd.concat(train_classes_df, axis=0).reset_index(drop=True)
    test_classes_df  = pd.concat(test_classes_df, axis=0).reset_index(drop=True)

    return train_classes_df, test_classes_df


def split_by_min_max_frequency(
        counts_df, 
        min_n_samples,
        max_n_samples,
    ):
    train_mask = ((counts_df['frequency'] <= max_n_samples) &
                  (counts_df['frequency'] >= min_n_samples))
    train_classes_df = counts_df[train_mask]
    test_classes_df  = counts_df[~train_mask]
    return train_classes_df, test_classes_df


if __name__ == '__main__':
    args = parse_args()

    dataset_info_path = Path(args.dataset_info)
    dataset_path = dataset_info_path.parents[0]

    df = pd.read_csv(
        dataset_info_path, 
        dtype={
            'label': str,
            'file_name': str,
            'width': int,
            'height': int,
            'hash': str,
        }
    )

    counts_df = get_counts_df(df)

    if args.split_type == 'minmax':
        train_classes_df, test_classes_df = split_by_min_max_frequency(
            counts_df, 
            min_n_samples=args.min_n_samples,
            max_n_samples=args.max_n_samples
        )
    
    elif args.split_type == 'freq':
        train_classes_df, test_classes_df = split_by_frequency_bins(
            counts_df, 
            test_ratio=args.test_ratio,
            min_frequency=args.min_freq,
            random_seed=args.random_seed
        )

    n_samples_total = counts_df['frequency'].sum()
    n_samples_train = train_classes_df['frequency'].sum()
    n_samples_test  = test_classes_df['frequency'].sum()

    print('Dataset info:')
    print(f'Number of classes: {len(counts_df)}')
    print(f'Number of samples: {n_samples_total}')
    print(counts_df.describe()) 

    print('\nTraining set:')
    print(f'Number of classes: {len(train_classes_df)}')
    print(f'Number of samples: {n_samples_train}')
    print(train_classes_df.describe()) 

    print('\nTesting set:')
    print(f'Number of classes: {len(test_classes_df)}')
    print(f'Number of samples: {n_samples_test}')
    print(test_classes_df.describe()) 

    train_classes_df.to_csv(dataset_path / 'classes_train.csv', index=False)
    test_classes_df.to_csv(dataset_path / 'classes_test.csv', index=False)

    dataset_train = df[df['label'].isin(train_classes_df['label'])].reset_index(drop=True)
    dataset_test = df[df['label'].isin(test_classes_df['label'])].reset_index(drop=True)

    dataset_train.to_csv(dataset_path / 'dataset_train.csv', index=False)
    dataset_test.to_csv(dataset_path / 'dataset_test.csv', index=False)

    print(dataset_train.head())
