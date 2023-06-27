import argparse
import pandas as pd
from pathlib import Path
from utils import add_image_sizes, download_dataset


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
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.download:
        save_path = Path(args.save_path)
        save_path.mkdir(exist_ok=True)
        download_dataset(save_path, dataset='inshop')
        dataset_path = save_path / 'inshop'
    else:
        dataset_path = Path(args.dataset_path)
    save_path = dataset_path

    df_info   = pd.read_csv(save_path / 'list_eval_partition.txt', 
                              sep='\s+', 
                              skiprows=1)
    df_info = df_info.rename(columns={'image_name' : 'file_name',
                                      'item_id' : 'label'})
    df_info = df_info.assign(is_test=[0 if x == 'train' else 1 for x in df_info['evaluation_status']])
    df_info = add_image_sizes(df_info, 
                              dataset_path)
    df_info = df_info[['file_name', 
                       'label', 
                       'width', 
                       'height',
                       'evaluation_status',
                       'is_test']]
    
    df_info.to_csv(save_path / 'dataset_info.csv', index=False) 

    