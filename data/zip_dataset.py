import shutil
import os
import argparse
from pathlib import Path


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
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    dataset_path = Path(args.dataset_path)
    save_path = Path(args.save_path)

    dataset_name = dataset_path.name
    print(f'Create zip archieve for {dataset_name} dataset')

    os.system(f"cd {dataset_path.parents[0]} && tar --use-compress-program=\"pigz -k \" -cf {dataset_name}.tar.gz {dataset_name}")
    