from imagededup.methods import PHash
from pathlib import Path
from tqdm.auto import tqdm
from collections import defaultdict
import argparse
import pandas as pd


# def filter_duplicates(df, threshold=10):
#     phasher = PHash(verbose=False)
#     total_n_samples = 0
#     total_n_duplicates = 0
#     duplicates_list = []
#     classes_dict = defaultdict(dict)

#     labels_groups = df.groupby('label')
#     for label, label_group in tqdm(labels_groups):
#         classes_dict[label] = pd.Series(
#             label_group.hash.values,
#             index=label_group.file_name
#         ).to_dict()
    
#     progress_bar = tqdm(classes_dict.items(), 
#                         total=len(classes_dict))

#     for cl_name, cl_imgs in progress_bar:
#         duplicates = phasher.find_duplicates_to_remove(
#             encoding_map=cl_imgs,
#             max_distance_threshold=threshold
#         )

#         for duplicate in duplicates:
#             duplicates_list.append(str(duplicate)+'\n')

#         total_n_samples += len(cl_imgs)
#         total_n_duplicates += len(duplicates)

#         progress_bar.set_postfix({
#             'Number of duplicates' : f'{total_n_duplicates}/{total_n_samples}'
#         })
    
#     duplicates_percentage = (total_n_duplicates / total_n_samples) * 100
#     print(f'Duplicates percentage: {duplicates_percentage}')
    
#     print(f'Before removing duplicates: {len(df)}')
#     df = df[~df['file_name'].isin(duplicates_list)]
#     print(f'After removing duplicates: {len(df)}')
#     return df


def filter_min_n_samples_per_class(df, min_n_samples=5):
    good_dfs = []

    labels_groups = df.groupby('label')
    print(f'N labels before: {len(labels_groups)}')

    for label, label_group in tqdm(labels_groups):
        if len(label_group)>=min_n_samples:
            good_dfs.append(label_group)

    print(f'N good labels: {len(good_dfs)}')
    df_filtered = pd.concat(good_dfs).reset_index(drop=True)
    
    return df_filtered


def undersampling(
        df, 
        max_n_samples=100,
        random_state=28
    ):
    train_groups = []
    for label, group in df.groupby('label'):
        group = group.reset_index(drop=True)
        if len(group)>max_n_samples:
            group = group.sample(
                max_n_samples, 
                random_state=random_state
            )
        train_groups.append(group)
    df = pd.concat(train_groups).reset_index(drop=True)
    return df



def parse_args():
    parser = argparse.ArgumentParser(description='filter dataset')
    parser.add_argument(
        '--dataset_info', 
        default="./", 
        help="path to the dataset info file"
    )
    parser.add_argument(
        '--filter_info', 
        default="./", 
        help="path to the filter file"
    )
    parser.add_argument(
        '--max_n_samples', 
        default=-1, 
        type=int,
        help="max number of samples per class"
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    dataset_info_path = Path(args.dataset_info)
    dataset_path = dataset_info_path.parents[0]
    filter_info_path = Path(args.filter_info)
    
    dataset_info_df = pd.read_csv(
        dataset_info_path, 
        dtype={
            'label': str,
            'file_name': str,
            'width': int,
            'height': int,
            'hash' : str
        }
    )

    filter_info_df = pd.read_csv(
        filter_info_path, 
        dtype={
            'upc': str,
            'uuid': str,
        }
    )

    filter_info_df['file_name'] = filter_info_df['upc'] + '/' + filter_info_df['uuid']
    good_samples = filter_info_df.file_name.tolist()

    print(filter_info_df)

    print('BEFORE file filtering')
    print(dataset_info_df)

    dataset_info_df = dataset_info_df[dataset_info_df['file_name'].isin(good_samples)]
    dataset_info_df.reset_index(
        inplace=True, 
        drop=True
    )

    print('AFTER file filtering')
    print(dataset_info_df)

    
    if args.max_n_samples > 0:
        print(f'Before max n samples filtering: {len(dataset_info_df)}')
        dataset_info_df = undersampling(dataset_info_df, max_n_samples=args.max_n_samples)
        print(f'After max n samples filtering: {len(dataset_info_df)}')

    dataset_info_df.to_csv(dataset_path / 'dataset_info_filtered.csv', index=False) 
    