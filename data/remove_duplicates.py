from imagededup.methods import PHash
from imagededup.utils import plot_duplicates
from pathlib import Path
from tqdm.auto import tqdm
import json
from collections import defaultdict
import random
import argparse
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='find dublicates')
    parser.add_argument('--dataset_file', default="./", help="path to the dataset file")
    parser.add_argument('--duplicates_file', default="./", help="path to the duplicates file")
    args = parser.parse_args()

    DATASET_CSV_PATH = Path(args.dataset_file)
    DATASET_CSV_NAME = DATASET_CSV_PATH.stem 
    DATASET_PATH = DATASET_CSV_PATH.parents[0]
    DUPLICATES_FILE = Path(args.duplicates_file)

    df = pd.read_csv(DATASET_CSV_PATH, dtype={'label': str,
                                              'file_name': str,
                                              'width': int,
                                              'height': int,
                                              'hash' : str})
    with open(DUPLICATES_FILE) as fp:
        duplicates = fp.readlines()
    duplicates = [dup.strip() for dup in duplicates]

    print(f'Before removing duplicates: {len(df)}')
    df = df[~df['file_name'].isin(duplicates)]
    print(f'After removing duplicates: {len(df)}')

    df.to_csv(DATASET_PATH / f'{DATASET_CSV_NAME}_dedup.csv', index=False)

    