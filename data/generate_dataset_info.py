import imagesize
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
import argparse
import multiprocessing


def parse_args():
    parser = argparse.ArgumentParser(description='generate dataset information file')
    parser.add_argument(
        '--dataset_path', 
        default="./", 
        help="path to the dataset"
    )
    parser.add_argument(
        '--hashes', 
        action='store_true', 
        help='calculate hashes'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    DATASET_PATH = Path(args.dataset_path)
    SIZES_FILE = DATASET_PATH / 'dataset_info.csv'
    
    CLASS_FOLDERS = [x for x in DATASET_PATH.glob('*/') if x.is_dir()]
    image_infos = []

    for class_idx, class_folder in tqdm(enumerate(CLASS_FOLDERS), 
                                        total=len( CLASS_FOLDERS)):
        cl_name = str(class_folder.name)
        for item in class_folder.iterdir():
            if item.is_file():
                try:
                    image_name = str(item.name)
                    width, height = imagesize.get(item)
                    image_infos.append([cl_name, image_name, width, height])
                except:
                    print(f'Problem with file: {str(item)}')

    df = pd.DataFrame(
        image_infos, 
        columns=[
            'label', 
            'file_name', 
            'width', 
            'height'
            ],
    )
    df['file_name'] = df['label'] + '/' + df['file_name']
    print(df.dtypes)
    print('Total N of images: ', len(df))

    if args.hashes:
        from imagededup.methods import PHash
        print('Generate hashes')
        phasher = PHash()
        n_workers = multiprocessing.cpu_count() 
        encodings = phasher.encode_images(image_dir=DATASET_PATH, 
                                          recursive=True,
                                          num_enc_workers=n_workers)
        df['hash'] = df['file_name'].map(encodings)
        print('Hashes were added to dataset info file')
        
    df.to_csv(DATASET_PATH / 'dataset_info.csv', index=False) 
