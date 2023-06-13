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
    parser.add_argument('--save_path', default="", help="result save path")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    save_path = Path(args.save_path)
    save_path.mkdir(exist_ok=True)

    df_info = pd.read_table(args.desc_csv, 
                                    delimiter=',',
                                    dtype={  
                                            "upc": str ,
                                            "retailer" : str,
                                            "name" : str,
                                            "category" : str,
                                            "product_image" : str
                                  })
    
    df_info = df_info[["upc",
                       "retailer",
                       "name",
                       "category" ]]
    
    df_info = df_info[df_info['upc'].notna()]
    df_info = df_info[df_info['name'].notna()]

    df_info['name'] = df_info['name'].str.lower()
    df_info['category'] = df_info['category'].str.lower()

    df_info = df_info[~df_info['name'].str.contains('product with code')]
    
    n_samples = len(df_info)
    n_classes = len(df_info['upc'].unique())

    print(f'Number of samples: {n_samples}')
    print(f'Number of classes: {n_classes}')

    print('Remove duplicates')
    duplicated_upcs = df_info['upc'][df_info['upc'].duplicated()].unique()
    print('Before: ', len(df_info))
    df_dup  = df_info[df_info['upc'].isin(duplicated_upcs)]
    df_info = df_info[~df_info['upc'].isin(duplicated_upcs)]
    dup_groups = df_dup.groupby(['upc'])
    print('After: ', len(df_info))

    dedup_rows     = []
    for dup_upc in tqdm(duplicated_upcs):
        df_dup = dup_groups.get_group(dup_upc)

        product_names     = df_dup.name.dropna().unique()
        product_cats      = df_dup.category.dropna().unique()
        product_retailers = df_dup.retailer.dropna().unique()
        
        combined_name     = '##'.join(product_names).lower()
        combined_cat      = '##'.join(product_cats).lower()
        combined_retailer = '##'.join(product_retailers).lower()

        dedup_rows.append((
            dup_upc, 
            combined_name,
            combined_cat,
            combined_retailer,
        ))

    df_dedup = pd.DataFrame(
        dedup_rows,
        columns =['upc', 
                  'name', 
                  'category',
                  'retailer'],
        dtype=str
    )

    print(df_dedup)

    df_info = df_info.append(df_dedup, ignore_index=True)
    print('Deduplicated products info size: ', len(df_info))

    print('After filtering:')
    n_samples = len(df_info)
    n_classes = len(df_info['upc'].unique())

    print(f'Number of samples: {n_samples}')
    print(f'Number of classes: {n_classes}')

    df_info.to_csv(save_path / f'upcs_infos_filtered.csv', index=False)

    