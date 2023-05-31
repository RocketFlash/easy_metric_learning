import imagesize
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='split dataset on training and testing parts')
    parser.add_argument('--dataset_csv', default="./", help="path to the dataset")
    parser.add_argument('--max_n_samples', type=int, default=50, help="max n of samples to select class for training")
    parser.add_argument('--min_n_samples', type=int, default=3, help="min n of samples to be in training set")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    DATASET_CSV = Path(args.dataset_csv)
    DATASET_PATH = DATASET_CSV.parents[0]
    THRESHOLD = args.max_n_samples
    MIN_N_SAMPLES = args.min_n_samples

    df = pd.read_csv(DATASET_CSV, dtype={
                                         'label': str,
                                         'file_name': str,
                                         'width': int,
                                         'height': int,
                                         'hash': str,
                                        })
    
    counts = df['label'].value_counts()
    counts_df = pd.DataFrame({
                              'label':counts.index, 
                              'frequency':counts.values
                             })
    n_samples_before = counts_df['frequency'].sum()

    print('Before filtering:')
    print(f'Number of classes : {len(counts_df)}')
    print(f'Number of samples: {n_samples_before}')

    counts_df = counts_df[counts_df['frequency']>=MIN_N_SAMPLES]
    n_samples_after = counts_df['frequency'].sum()

    print(f'After filtering (n_samples >= {MIN_N_SAMPLES}):')
    print(f'Number of classes : {len(counts_df)}')
    print(f'Number of samples: {n_samples_after}')

    df = df[df['label'].isin(counts_df['label'])].reset_index(drop=True)

    df.to_csv(DATASET_PATH / 'dataset_all.csv', index=False)
    counts_df.to_csv(DATASET_PATH / 'classes_all.csv', index=False)

    train_df = counts_df[counts_df['frequency'] <= THRESHOLD]
    test_df = counts_df[counts_df['frequency'] > THRESHOLD]
    n_samples_train = train_df['frequency'].sum()
    n_samples_test = test_df['frequency'].sum()

    print('Training set:')
    print(f'Number of classes: {len(train_df)}')
    print(f'Number of samples: {n_samples_train}')
    print(train_df.describe()) 

    print('Testing set:')
    print(f'Number of classes: {len(test_df)}')
    print(f'Number of samples: {n_samples_test}')
    print(test_df.describe()) 

    train_df.to_csv(DATASET_PATH / 'classes_train.csv', index=False)
    test_df.to_csv(DATASET_PATH / 'classes_test.csv', index=False)

    dataset_train = df[df['label'].isin(train_df['label'])].reset_index(drop=True)
    dataset_test = df[df['label'].isin(test_df['label'])].reset_index(drop=True)

    dataset_train.to_csv(DATASET_PATH / 'dataset_train.csv', index=False)
    dataset_test.to_csv(DATASET_PATH / 'dataset_test.csv', index=False)

    print(f'Number of samples train : {len(dataset_train)}')
    print(f'Number of samples test : {len(dataset_test)}')

    print(dataset_train.head())
