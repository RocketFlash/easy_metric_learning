import argparse
import pandas as pd
from pathlib import Path
import shutil
from tqdm.auto import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='find dublicates')
    parser.add_argument('--dataset_path', default="./", help="path to the dataset")
    parser.add_argument('--ann_path', default="./", help="path to the annotation file")
    parser.add_argument('--save_path', default="~/tmp", help='save directory path')
    parser.add_argument('--filename_col', default="filename", help="name of filename column")
    parser.add_argument('--label_col', default="label", help="name of label column")
    parser.add_argument('--ext', default=".jpg", help="images extensions")
    return parser.parse_args()


if __name__ == '__main__':
    args  = parse_args()
    ann_file_path = Path(args.ann_path)
    dataset_path  = Path(args.dataset_path)
    save_path     = Path(args.save_path)
    save_path.mkdir(exist_ok=True)
    f_c, l_c = args.filename_col, args.label_col

    if ann_file_path.suffix == '.xlsx':
        df = pd.read_excel(args.ann_path, dtype={
                                                 f_c: str,
                                                 l_c: str,
                                                }) 
    else:
        df = pd.read_csv(args.ann_path, dtype={
                                               f_c: str,
                                               l_c: str,
                                              }) 

    for index, row in tqdm(df.iterrows(), total=len(df)):
        file_name, label = row[f_c], row[l_c]
        save_path_label = save_path / label
        
        if not save_path_label.is_dir():
            save_path_label.mkdir()

        image_name = file_name + args.ext
        image_path = dataset_path / image_name
        shutil.copy(image_path, save_path_label)