from pathlib import Path
import cv2
import functools
import multiprocessing as mp
import pandas as pd
import argparse
from tqdm.auto import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='get dataset information file')
    parser.add_argument('--dataset_path', default="./", help="path to the dataset")
    parser.add_argument('--dataset_csv', default="./", help="path to the dataset info file")
    parser.add_argument('--image_size', type=int, default=170, help="minimal image size")
    parser.add_argument('--save_path', default="./", help="save path")
    return parser.parse_args()


def resize_image(image_path, dataset_path='', image_size=(170,170), save_path='./'):
    image = cv2.imread(str(dataset_path/image_path))
    image = cv2.resize(image, image_size)
    cv2.imwrite(str(save_path/image_path), image)


if __name__ == '__main__':
    args = parse_args()

    dataset_path = Path(args.dataset_path)
    save_path    = Path(args.save_path)
    save_path.mkdir(exist_ok=True)
    image_size = (args.image_size, args.image_size)
    n_cpu = mp.cpu_count() 
    print(f'N CPUs: {n_cpu}')

    df = pd.read_csv(args.dataset_csv, dtype={
                                              'label': str,
                                              'file_name': str,
                                              'width': int,
                                              'height': int,
                                              'hash' : str
                                              })
    class_names = df.label.unique()
    for class_name in class_names:
        class_folder = save_path / class_name
        class_folder.mkdir(exist_ok=True)

    df['width'] = image_size[0]
    df['height'] = image_size[1]

    images_paths = []
    for index, row in df.iterrows():
        images_paths.append(row['file_name'])

    resize_image_fn = functools.partial(resize_image,
                                        dataset_path=dataset_path,
                                        image_size=image_size,
                                        save_path=save_path)
    
    pool = mp.Pool(n_cpu)

    progress_bar = tqdm(pool.imap(resize_image_fn, images_paths), total=len(images_paths))

    pool.close()
    pool.join()

    df.to_csv(save_path / 'dataset_info.csv', index=False)



    