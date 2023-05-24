import pandas as pd
from pathlib import Path


if __name__ == '__main__':
    DATASET_CSV_PATH_1 = Path('/home/rauf/datasets/metric_learning/fv_mksp/dataset_info.csv')
    DATASET_CSV_PATH_2 = Path('/home/rauf/datasets/metric_learning/mksp/dataset_info.csv')
    DATASET_CSV_PATH_3 = Path('/home/rauf/datasets/metric_learning/mksp_full/dataset_info.csv')

    df1 = pd.read_csv(DATASET_CSV_PATH_1, dtype={'label': str,
                                              'file_name': str,
                                              'width': int,
                                              'height': int,
                                              'hash' : str})
    
    df2 = pd.read_csv(DATASET_CSV_PATH_2, dtype={'label': str,
                                              'file_name': str,
                                              'width': int,
                                              'height': int,
                                              'hash' : str})
    
    df3 = pd.read_csv(DATASET_CSV_PATH_3, dtype={'label': str,
                                              'file_name': str,
                                              'width': int,
                                              'height': int,
                                              'hash' : str})
    
    print(f'N samples dataset1: {len(df1)}')
    print(f'N samples dataset2: {len(df2)}')
    print(f'N samples dataset2: {len(df3)}')
    
    classes1 = set(df1['label'].unique())
    classes2 = set(df2['label'].unique())
    classes3 = set(df3['label'].unique())

    print(f'N classes dataset1: {len(classes1)}')
    print(f'N classes dataset2: {len(classes2)}')
    print(f'N classes dataset2: {len(classes3)}')

    similar_classes = classes2.intersection(classes1)
    print(f'N similar classes : {len(similar_classes)}')