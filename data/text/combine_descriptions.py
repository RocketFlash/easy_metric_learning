import pandas as pd
from pathlib import Path
import json
import argparse
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='analyze dataset descriptions')
    parser.add_argument('--descriptions_path', default="", help="path to the csv descriptions folder")
    parser.add_argument('--filter', action='store_true', help='remove duplicates')
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

    if args.filter:
        df_info = df_combined[[ "upc",
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


        