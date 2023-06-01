import pandas as pd
import argparse
import numpy as np
import pprint
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='search ')
    parser.add_argument('--work_dirs', type=str, default='work_dirs/', help='path to work_dirs')
    parser.add_argument('--dataset_name', type=str, default='product_recognition', help='dataset_name')
    parser.add_argument('--save_path', type=str, default='results', help='results save path')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    work_dirs = Path(args.work_dirs)
    dataset_name = args.dataset_name

    save_path = Path(args.save_path)
    save_path.mkdir(exist_ok=True, 
                    parents=True)
    
    runs_paths = [f for f in work_dirs.iterdir() if f.is_dir()]
    all_metrics = []

    for run_path in runs_paths:
        run_name = run_path.name
        embeddings_folder = run_path / 'embeddings'
        embeddings_dataset = embeddings_folder / dataset_name
        metrics_csv_path = embeddings_dataset / 'metrics.csv'
        if metrics_csv_path.is_file():
            df = pd.read_csv(metrics_csv_path)
            df = df.set_index('metric').T
            df.insert(0, 'run', run_name)
            df = df.set_index('run')
            all_metrics.append(df)
    
    all_metrics = pd.concat(all_metrics)
    print(all_metrics)
    all_metrics.to_csv(save_path / f'{dataset_name}_metrics.csv')
    
