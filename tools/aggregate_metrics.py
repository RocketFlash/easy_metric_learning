import pandas as pd
import argparse
import numpy as np
import pprint
from pathlib import Path
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description='search ')
    parser.add_argument('--work_dirs', type=str, default='work_dirs/test/', help='path to work_dirs')
    parser.add_argument('--csv_file_name', type=str, default='eval_result_on_retail.csv', help='evaluation results csv file')
    parser.add_argument('--save_path', type=str, default='results/dataset_metrics', help='results save path')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    work_dirs = Path(args.work_dirs)
    save_path = Path(args.save_path)
    save_path.mkdir(exist_ok=True, parents=True)
    
    runs_paths = [f for f in work_dirs.iterdir() if f.is_dir()]
    all_datasets_metrics = defaultdict(lambda: defaultdict(dict))

    for run_path in runs_paths:
        run_name = run_path.name
        csv_file_path = run_path / args.csv_file_name
        
        if csv_file_path.is_file():
            df = pd.read_csv(csv_file_path, index_col=0)
            metric_names = list(df.columns)
            
            for dataset_name, metrics in df.iterrows():
                for metric_name in metric_names:
                    all_datasets_metrics[dataset_name][run_name][metric_name] = metrics[metric_name]


    for dataset_name, dataset_metrics in all_datasets_metrics.items():
        df_dataset_metrics = pd.DataFrame.from_dict(dataset_metrics, orient='index')
        df_dataset_metrics = df_dataset_metrics.sort_values(by='R@1', ascending = False)
        df_dataset_metrics.to_csv(save_path / f'{dataset_name}_metrics.csv')
    
        
    
    
