from pathlib import Path
import shutil

if __name__ == '__main__':
    DATASET_PATH_1 = Path('/home/rauf/datasets/metric_learning/fv_mksp/')
    DATASET_PATH_2 = Path('/home/rauf/datasets/metric_learning/mksp/')
    SAVE_PATH = Path('/home/rauf/datasets/metric_learning/mksp_full')
    SAVE_PATH.mkdir(exist_ok=True)

    shutil.copytree(DATASET_PATH_1, SAVE_PATH, dirs_exist_ok=True)
    shutil.copytree(DATASET_PATH_2, SAVE_PATH, dirs_exist_ok=True)