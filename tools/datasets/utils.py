import imagesize
import numpy as np
import zipfile
import gdown
import wget
import tarfile
import shutil


def download_dataset(save_path, dataset='inshop'):
    if dataset=='inshop':
        # File ids from InShop dataset: https://drive.google.com/drive/folders/0B7EVK8r0v71pVDZFQXRsMDZCX1E?resourcekey=0-4R4v6zl4CWhHTsUGOsTstw
        IMAGES_URL = '0B7EVK8r0v71pS2YxRE1QTFZzekU'
        SPLIT_URL  = '0B7EVK8r0v71pYVBqLXpRVjhHeWM'

        print(f'Downloading dataset...')
        dataset_path = save_path / 'inshop'
        dataset_path.mkdir(exist_ok=True)

        z_file_path     = dataset_path / 'img.zip'
        split_file_path = dataset_path / 'list_eval_partition.txt'

        gdown.download(id=IMAGES_URL, 
                    output=str(z_file_path), 
                    quiet=False)

        with zipfile.ZipFile(z_file_path, 'r') as z_f:
            z_f.extractall(str(dataset_path))

        z_file_path.unlink()

        gdown.download(id=SPLIT_URL, 
                    output=str(split_file_path), 
                    quiet=False)
        print(f'Dataset was downloaded and extracted')
    elif dataset=='sop':
        DATASET_URL = 'ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip'
        print(f'Download dataset from : {DATASET_URL}')

        wget.download(DATASET_URL, out=str(save_path))
        z_file = save_path / 'Stanford_Online_Products.zip'
        with zipfile.ZipFile(z_file, 'r') as z_f:
            z_f.extractall(str(save_path))
        z_file.unlink()
        print(f'Dataset was downloaded and extracted')
    elif dataset=='cub':
        DATASET_URL = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1'
        print(f'Download dataset from : {DATASET_URL}')

        save_tar_file = save_path / 'CUB_200_2011.tgz'
        wget.download(DATASET_URL, out=str(save_tar_file))

        with tarfile.open(save_tar_file) as tar:
            tar.extractall(save_path)

        ds_path = save_path / 'CUB_200_2011'
        shutil.move(save_path / 'attributes.txt',
                    ds_path   / 'attributes.txt')

        save_tar_file.unlink()
        print(f'Dataset was downloaded and extracted')
    else:
        print('Unknown dataset')




def add_image_sizes(df, dataset_path, with_images_folder=False):
    image_sizes = []

    images_folder = dataset_path
    if with_images_folder:
        images_folder = images_folder / 'images'

    for index, row in df.iterrows():
        image_path = images_folder / row.file_name
        width, height = imagesize.get(image_path)
        image_sizes.append([width, height])

    image_sizes = np.array(image_sizes)
    df['width']  = list(image_sizes[:, 0])
    df['height'] = list(image_sizes[:, 1])
    return df