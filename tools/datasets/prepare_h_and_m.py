import sys
sys.path.append("./")

import argparse
import pandas as pd
from pathlib import Path
from utils import add_image_sizes, download_dataset
from data.utils import get_stratified_kfold


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', 
                        type=str, 
                        default='./', 
                        help='dataset path')
    parser.add_argument('--save_path', 
                        type=str, 
                        default='./', 
                        help='save path')
    parser.add_argument('--download', 
                        action='store_true', 
                        help='Dowload images')
    parser.add_argument('--random_seed', 
                        default=28, 
                        help='random seed')
    return parser.parse_args()
    

if __name__ == '__main__':
    args = parse_args()

    if args.download:
        '''
        In order to download H&M dataset you must have kaggle account 
        and generated kaggle.json file in ~/.kaggle/kaggle.json 
        '''
        
        save_path = Path(args.save_path)
        save_path.mkdir(exist_ok=True)
        download_dataset(save_path, dataset='h_and_m')
        dataset_path = save_path / 'h_and_m'
    else:
        dataset_path = Path(args.dataset_path)
    save_path = dataset_path

    df_info  = pd.read_csv(dataset_path / 'articles.csv')
    product_code_unique = df_info.product_code.unique()
    mapping_table = {}
    for n, i in enumerate(product_code_unique):
        mapping_table[i] = n
    df_info.product_code = df_info.product_code.map(lambda x: mapping_table[x])
    l_f = lambda x: (dataset_path / f'images/{x}').is_file()
    l_f_n = lambda x: f'0{str(x)[:2]}/0{str(x)}.jpg'
    df_info['file_name'] = df_info.article_id.map(l_f_n)
    df_info['avaliable'] = df_info.file_name.map(l_f)
    df_info.drop(df_info[df_info.avaliable == False].index, inplace = True)

    df_info = df_info.rename(columns={'article_id' : 'label',
                                      'detail_desc': 'title'})
    df_info = df_info[['file_name', 'label' , 'title']]
    df_info['label'] = df_info['label'].apply(lambda x: f'h_and_m_{x}')
    df_info.reset_index(drop=True, inplace=True)
    df_info['is_test'] = 0
    
    df_info = add_image_sizes(df_info, 
                              dataset_path,
                              with_images_folder=True)
    
    print(df_info)

    df_info.to_csv(save_path / 'dataset_info.csv', index=False) 

    