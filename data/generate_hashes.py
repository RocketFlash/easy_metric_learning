from imagededup.methods import PHash
from imagededup.utils import plot_duplicates
from pathlib import Path
from tqdm.auto import tqdm
import json
from collections import defaultdict
import random
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='find dublicates')
    parser.add_argument('--dataset_path', default="./", help="path to the dataset")
    args = parser.parse_args()

    DATASET_PATH = Path(args.dataset_path)
    HASHES_PATH = DATASET_PATH / 'hashes.json'
    phasher = PHash()

    if HASHES_PATH.is_file():
        print('Hashes file already exist')
    else:
        print(f'Generate encodings and save them in {str(HASHES_PATH)}')
        encodings = phasher.encode_images(image_dir=DATASET_PATH, recursive=True)
        with open(HASHES_PATH, 'w') as fp:
            json.dump(encodings, fp)
        print(f'Encodings are ready!')
    