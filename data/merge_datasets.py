import pandas as pd
import argparse
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='filter dataset')
    parser.add_argument('--dataset_1_info', default="./", help="path to the dataset1 info file")
    parser.add_argument('--dataset_2_info', default="./", help="path to the dataset2 info file")
    parser.add_argument('--min_size', type=int, default=-1, help="minimal image size")
    parser.add_argument('--max_size', type=int, default=-1, help="maximal image size")
    parser.add_argument('--dedup', action='store_true', help='remove duplicates')
    parser.add_argument('--threshold', type=int, default=5, help="threshold for duplicates selection")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    df1 = pd.read_csv(args.dataset_1_info, 
                      dtype={  
                             'label': str,
                             'file_name': str,
                             'width': int,
                             'height': int,
                             'hash' : str
                             })
    df2 = pd.read_csv(args.dataset_2_info, 
                      dtype={  
                             'label': str,
                             'file_name': str,
                             'width': int,
                             'height': int,
                             'hash' : str
                             })
    
    labels_1 = df1.label.unique()
    labels_2 = df2.label.unique()

    print('N labels dataset1 : ', len(labels_1))
    print('N labels dataset2 : ', len(labels_2))

    labels_1_set = set(labels_1)
    labels_2_set = set(labels_2)

    labels_intersect = labels_1_set.intersection(labels_2_set)
    print('N intersected labels: ', len(labels_intersect))

    labels_2_unique = labels_2_set.difference(labels_1_set)
    print('N unique labels in dataset2: ', len(labels_2_unique))

    df2_unique = df2[df2['label'].isin(labels_2_unique)].reset_index(drop=True)
    print('Dataset2 unique rows: ', df2_unique)

    df2 = df2[~df2['label'].isin(labels_2_unique)].reset_index(drop=True)
    print('Dataset2 non unique rows: ', df2)

    labels2_groups = df2.groupby('label')
    for label, label_group in tqdm(labels_groups):
        pass