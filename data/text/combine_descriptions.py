import pandas as pd
from pathlib import Path
import json
import argparse
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='analyze dataset descriptions')
    parser.add_argument('--descriptions_path', default="", help="path to the csv descriptions folder")
    parser.add_argument('--save_path', default="folds", help="name of saved files")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    desc_path = Path(args.descriptions_path)
    desc_files = list(desc_path.glob('*.csv'))
    save_path = Path(args.save_path)
    save_path.mkdir(exist_ok=True)

    df_infos = []
    for desc_file in tqdm(desc_files):
        retailer_name = desc_file.stem.split('_')[0]

        df_descriptions = pd.read_table(desc_file, 
                                        delimiter=',',
                                        dtype=str)
        
        df_info = df_descriptions[['name',  
                                   'category_name',
                                   'primary_code',
                                   'primary_image']]
        
        df_info = df_info.rename(columns={'primary_code': 'upc',
                                          'category_name' : 'category',
                                          'primary_image': 'product_image' })
        df_info['retailer'] = retailer_name

        df_infos.append(df_info)

    df_combined = pd.concat(df_infos, ignore_index=True)
    df_combined.to_csv(save_path / f'upcs_infos.csv', index=False)


    