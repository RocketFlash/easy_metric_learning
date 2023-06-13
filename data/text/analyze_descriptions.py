import pandas as pd
from pathlib import Path
import json
import argparse
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='analyze dataset descriptions')
    parser.add_argument('--dataset_csv', default="", help="path to the dataset info csv file")
    parser.add_argument('--desc_csv', default="", help="path to the descriptions csv file")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    dataset_csv = Path(args.dataset_csv)

    df_dataset = pd.read_table(dataset_csv, 
                               delimiter=',',
                               dtype={  
                                         'label': str,
                                         'file_name': str,
                                         'width': int,
                                         'height': int,
                                         'hash' : str
                                })
    
    print('Dataset info size: ', len(df_dataset))

    df_info = pd.read_table(args.desc_csv, 
                                    delimiter=',',
                                    dtype={  
                                            "upc": str ,
                                            "retailer" : str,
                                            "name" : str,
                                            "category" : str,
                                  })
    
    df_dataset = df_dataset.merge(df_info, 
                                  how='inner', 
                                  left_on='label', 
                                  right_on='upc',
                                  validate='many_to_one')

    print('Dataset info size + categories: ', len(df_dataset))
    
    
    n_samples = len(df_dataset)
    n_cats    = len(df_dataset['category'].unique())

    print(f'Number of samples: {n_samples}')
    print(f'Number of categories: {n_cats}')

    print(df_dataset.category.value_counts())

    save_name = dataset_csv.stem + '_categories.csv'
    save_path = dataset_csv.parents[0] / save_name
    
    df_dataset.to_csv(save_path, index=False)


    
    