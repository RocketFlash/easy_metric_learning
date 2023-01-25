import imagesize
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
import argparse
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='get dataset information file')
    parser.add_argument('--dataset_path', default="./", help="path to the dataset")
    parser.add_argument('--hashes_path', default=None, help="path to the hases json file")
    parser.add_argument('--min_size', type=int, default=100, help="minimal image size")
    args = parser.parse_args()

    DATASET_PATH = Path(args.dataset_path)
    HASHES_PATH = args.hashes_path
    MIN_SIZE = args.min_size
    SIZES_FILE = DATASET_PATH / 'dataset_info.csv'

    if SIZES_FILE.is_file():
        print('Load sizes info file...')
        df = pd.read_csv(SIZES_FILE, dtype={'label': str,
                                        'file_name': str,
                                        'width': int,
                                        'height': int
                                    })
    else:
        class_folders = [x for x in DATASET_PATH.glob('*/')]
        all_sizes = []

        for class_folder in tqdm(class_folders):
            cl_name = class_folder.name
            for item in class_folder.iterdir():
                if item.is_file():
                    try:
                        image_name = item.name
                        width, height = imagesize.get(item)
                        all_sizes.append([cl_name, image_name, width, height])
                    except:
                        print(f'Problem with file: {str(item)}')

        df = pd.DataFrame(all_sizes, columns = ['label', 'file_name', 'width', 'height'])
        df.to_csv(DATASET_PATH / 'dataset_info.csv', index=False) 

    corrupted_images = df[(df['width']<MIN_SIZE) | (df['height']<MIN_SIZE)]
    corrupted_indexes = []
    
    for index, row in corrupted_images.iterrows():
        corrupted_file_path = DATASET_PATH / row['label'] / row['file_name']
        corrupted_indexes.append(index)

    print(f'Total number of removed images: {len(corrupted_indexes)}')
    df_filtered = df.drop(corrupted_indexes, axis=0)
    df_filtered['file_name'] = df_filtered['label'] + '/' + df_filtered['file_name']
     
        
    print(df_filtered.describe())
    if HASHES_PATH is not None:
        print('Load hashes')
        with open(HASHES_PATH, 'r') as fp:
            encodings = json.load(fp)
        df_filtered['hash'] = df_filtered['file_name'].map(encodings)
        print('Hashes were added to dataset info file')

    df_filtered.to_csv(DATASET_PATH / 'dataset_info_filtered.csv', index=False)
