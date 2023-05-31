import pandas as pd
from pathlib import Path
import json
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='analyze dataset using UPCs map')
    parser.add_argument('--dataset_csv', default="", help="path to the csv file describing dataset")
    parser.add_argument('--upcs_csv', default="", help="path to the UPCs description csv file")
    parser.add_argument('--save_name', default="folds", help="name of saved files")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # DATASET_CSV = Path(args.dataset_csv)
    # DATASET_PATH = DATASET_CSV.parents[0]

    # df = pd.read_csv(DATASET_CSV, dtype={
    #                                         'label': str,
    #                                         'file_name': str,
    #                                         'width': int,
    #                                         'height': int
    #                                     })
    
    # counts = df['label'].value_counts()
    # counts_df = pd.DataFrame({
    #                           'label':counts.index, 
    #                           'frequency':counts.values
    #                          })

    df_upcs = pd.read_table(args.upcs_csv, 
                            delimiter=',', 
                            dtype={  
                                    "PLANOGRAM TITLE": str ,
                                    "PRODUCT NAME" : str,
                                    "UPC" : str,
                                    "SHORT NAME" : str
                                  })
    
    df_upcs = df_upcs[['UPC', 'PRODUCT NAME', 'SHORT NAME']]
    print(df_upcs['PRODUCT NAME'].tolist())
    
    n_samples = len(df_upcs)
    classes = df_upcs['UPC'].unique()
    n_classes = len(classes)

    print(f'Number of samples: {n_samples}')
    print(f'Number of classes: {n_classes}')

    # counts_df['description'] = counts_df['label'].map(df_upcs.set_index('ean')['name'])
    # print(counts_df)

    