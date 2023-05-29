from imagededup.methods import PHash
from pathlib import Path
from tqdm.auto import tqdm
from collections import defaultdict
import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='filter dataset')
    parser.add_argument('--dataset_info', default="./", help="path to the dataset info file")
    parser.add_argument('--min_size', type=int, default=-1, help="minimal image size")
    parser.add_argument('--max_size', type=int, default=-1, help="maximal image size")
    parser.add_argument('--dedup', action='store_true', help='remove duplicates')
    parser.add_argument('--threshold', type=int, default=1, help="threshold for duplicates selection")
    return parser.parse_args()


def min_size_filtering(df, min_size=100):
    corrupted_images = df[(df['width']<min_size) | (df['height']<min_size)]
    corrupted_indexes = corrupted_images.index.tolist()
    print(f'Total number of removed images: {len(corrupted_indexes)}')
    return df.drop(corrupted_indexes, axis=0)


def max_size_filtering(df, max_size=1000):
    corrupted_images = df[(df['width']>max_size) | (df['height']>max_size)]
    corrupted_indexes = corrupted_images.index.tolist()
    print(f'Total number of removed images: {len(corrupted_indexes)}')
    return df.drop(corrupted_indexes, axis=0)


def duplicates_removal_filtering(df):
    phasher = PHash(verbose=False)
    total_n_samples = 0
    total_n_duplicates = 0
    duplicates_list = []
    classes_dict = defaultdict(dict)

    labels_groups = df.groupby('label')
    for label, label_group in tqdm(labels_groups):
        classes_dict[label] = pd.Series(label_group.hash.values,
                                        index=label_group.file_name).to_dict()
    
    progress_bar = tqdm(classes_dict.items(), 
                        total=len(classes_dict))

    for cl_name, cl_imgs in progress_bar:
        duplicates = phasher.find_duplicates_to_remove(encoding_map=cl_imgs,
                                                       max_distance_threshold=THRESHOLD)

        for duplicate in duplicates:
            duplicates_list.append(str(duplicate)+'\n')

        total_n_samples += len(cl_imgs)
        total_n_duplicates += len(duplicates)

        progress_bar.set_postfix({
                                    'Number of duplicates' : f'{total_n_duplicates}/{total_n_samples}'
                                })
    
    duplicates_percentage = (total_n_duplicates / total_n_samples) * 100
    print(f'Duplicates percentage: {duplicates_percentage}')
    
    print(f'Before removing duplicates: {len(df)}')
    df = df[~df['file_name'].isin(duplicates_list)]
    print(f'After removing duplicates: {len(df)}')
    return df


if __name__ == '__main__':
    args = parse_args()

    DATASET_CSV_PATH = Path(args.dataset_info)
    DATASET_CSV_NAME = DATASET_CSV_PATH.stem 
    DATASET_PATH = DATASET_CSV_PATH.parents[0]
    THRESHOLD = args.threshold

    df = pd.read_csv(DATASET_CSV_PATH, dtype={'label': str,
                                              'file_name': str,
                                              'width': int,
                                              'height': int,
                                              'hash' : str})

    if args.min_size > 0:
        df = min_size_filtering(df, min_size=args.min_size)

    if args.max_size > 0:
        df = max_size_filtering(df, max_size=args.max_size)
    
    if args.dedup:
        df = duplicates_removal_filtering(df)

    print(df.describe())
    df.to_csv(DATASET_PATH / 'dataset_info_filtered.csv', index=False)