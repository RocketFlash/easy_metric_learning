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
    parser.add_argument('--threshold', type=int, default=10, help="threshold for duplicates selection")
    args = parser.parse_args()

    DATASET_CSV_PATH = Path(args.dataset_file)
    DATASET_PATH = DATASET_CSV_PATH.parents[0]
    THRESHOLD = args.threshold

    phasher = PHash(verbose=False)

    df = pd.read_csv(DATASET_CSV_PATH, dtype={'label': str,
                                              'file_name': str,
                                              'width': int,
                                              'height': int,
                                              'hash' : str})
    
    total_n_samples = 0
    duplicates_list = []
    labels_ = df.label.unique()
    classes_dict = defaultdict(dict)

    for label in tqdm(labels):
        for index, row in df[df['label'] == label].iterrows():
            file_name = row['file_name']
            hash_v = row['hash']
            classes_dict[label][file_name] = hash_v
    
    for cl_name, cl_imgs in tqdm(classes_dict.items(), total=len(classes_dict)):
        duplicates = phasher.find_duplicates_to_remove(encoding_map=cl_imgs,
                                                       max_distance_threshold=THRESHOLD)

        for duplicate in duplicates:
            duplicates_list.append(str(duplicate)+'\n')

        total_n_samples += len(cl_imgs)
        duplicates_percentage = (len(duplicates) / total_n_samples) * 100

        print(f'Total number of duplicates: {total_n_duplicates}/{total_n_samples}')
        print(f'Duplicates percentage: {duplicates_percentage}')
    
    with open(DATASET_PATH / f'duplicates_t_{THRESHOLD}.txt', 'w') as f:
        f.writelines(duplicates_list)