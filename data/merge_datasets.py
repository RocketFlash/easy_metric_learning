import pandas as pd
import argparse
from tqdm import tqdm
from imagededup.methods import PHash


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
    labels_2_unique  = labels_2_set.difference(labels_1_set)

    print('N intersected labels: ', len(labels_intersect))
    print('N unique labels in dataset2: ', len(labels_2_unique))

    df2_unique = df2[df2['label'].isin(labels_2_unique)].reset_index(drop=True)
    df2 = df2[~df2['label'].isin(labels_2_unique)].reset_index(drop=True)
    
    print('Dataset2 unique rows: ', len(df2_unique))
    print('Dataset2 non unique rows: ', len(df2))

    phasher = PHash()

    df1['dataset'] = 1
    df2['dataset'] = 2

    # labels_groups_2 = df2.groupby('label')
    # for label, label_group_2 in tqdm(labels_groups_2):
    #     for index, row1 in label_group_2.iloc[:-1].iterrows():
    #         for index, row2 in label_group_2.iloc[index+1:].iterrows():
    #             distance = phasher.hamming_distance(row1['hash'], row2['hash'])
    #             if distance == 0:
    #                 print(row1['file_name'])
    #                 print(row2['file_name'])
    #                 print('===========')

    labels_groups_1 = df1.groupby('label')
    labels_groups_2 = df2.groupby('label')
    label_groups_combined = []
    for label, label_group_2 in tqdm(labels_groups_2):
        label_group_1 = labels_groups_1.get_group(label)

        label_group_combined = []
        row_indexes_to_remove = []
        
        for index1, row1 in label_group_1.iterrows():
            for index2, row2 in label_group_2.iterrows():
                distance = phasher.hamming_distance(row1['hash'], row2['hash'])
                if distance < args.threshold:
                    if row1['height'] < row2['height'] or row1['width'] < row2['width']:
                        label_group_combined.append(row2)
                        row_indexes_to_remove.append(index2)
                    else:
                        label_group_combined.append(row1)
                else:
                    label_group_combined.append(row1)
        if row_indexes_to_remove:
            label_group_2.drop(row_indexes_to_remove, inplace=True)
        
        for index2, row2 in label_group_2.iterrows():
            label_group_combined.append(row2)

        label_group_combined = pd.DataFrame(label_group_combined)
        label_groups_combined.append(label_group_combined)
        print(label_group_combined)
        
