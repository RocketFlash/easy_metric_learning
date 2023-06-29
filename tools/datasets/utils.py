import imagesize
import numpy as np
import zipfile
import gdown
import wget
import tarfile
import shutil
import os


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
        print(f'Dataset have downloaded and extracted')
    elif dataset=='sop':
        DATASET_URL = 'ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip'
        print(f'Download dataset from : {DATASET_URL}')

        wget.download(DATASET_URL, out=str(save_path))
        z_file = save_path / 'Stanford_Online_Products.zip'
        with zipfile.ZipFile(z_file, 'r') as z_f:
            z_f.extractall(str(save_path))
        z_file.unlink()
        print(f'Dataset have downloaded and extracted')
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
        print(f'Dataset have downloaded and extracted')
    elif dataset=='shopee':
        # install kaggle python package first and prepare kaggle.json file and put it in ~/.kaggle/kaggle.json
        dataset_path = save_path / 'shopee'
        dataset_path.mkdir(exist_ok=True)

        try:
            print(f'Downloading dataset...')
            os.system(f"kaggle competitions download -c shopee-product-matching -p {str(dataset_path)}")

            z_file = dataset_path / 'shopee-product-matching.zip'
            with zipfile.ZipFile(z_file, 'r') as z_f:
                z_f.extractall(str(dataset_path))

            z_file.unlink()
            print(f'Dataset have downloaded and extracted')
        except:
            print('In order to download Shopee dataset you must have kaggle account and generated kaggle.json file in ~/.kaggle/kaggle.json ')
    
    elif dataset=='met':
        IMAGES_URL = 'http://ptak.felk.cvut.cz/met/dataset/MET.tar.gz'
        dataset_path = save_path / 'met'
        dataset_path.mkdir(exist_ok=True)

        print(f'Download images from : {IMAGES_URL}')
        save_tar_file = dataset_path / 'MET.tar.gz'
        wget.download(IMAGES_URL, out=str(save_tar_file))

        with tarfile.open(save_tar_file) as tar:
            tar.extractall(dataset_path)

        save_tar_file.unlink()

        ANNOTATIONS_URL = 'http://ptak.felk.cvut.cz/met/dataset/ground_truth.tar.gz'
        print(f'Download annotations from : {ANNOTATIONS_URL}')
        save_tar_file = dataset_path / 'ground_truth.tar.gz'
        wget.download(ANNOTATIONS_URL, out=str(save_tar_file))

        with tarfile.open(save_tar_file) as tar:
            tar.extractall(dataset_path)

        save_tar_file.unlink()

        TEST_IMAGES_URL = 'http://ptak.felk.cvut.cz/met/dataset/test_met.tar.gz'
        print(f'Download test images from : {TEST_IMAGES_URL}')
        save_tar_file = dataset_path / 'test_met.tar.gz'
        wget.download(TEST_IMAGES_URL, out=str(save_tar_file))

        with tarfile.open(save_tar_file) as tar:
            tar.extractall(dataset_path)

        save_tar_file.unlink()

        print(f'Dataset have downloaded and extracted')
    elif dataset=='products10k':
        dataset_path = save_path / 'products10k'
        dataset_path.mkdir(exist_ok=True)

        TRAIN_IMAGES_URL = 'https://hxppaq.bl.files.1drv.com/y4mRRNNq8uUa-jR4FBBllPtxas1R00_ytt5IIXPFIWVZfxbBndfVZRRUebeWs9nWE3aowktixlQsXNZhFes-Cr_P26suWxEAA72YK1AsvNMSbqpxunzqxtGoPOanyS6xVM3lRDg0kol8HljzHnQ3rgJTmwb4qEX5g_TBoCvgE2bX7RdX-zWt1JnIDeqQrJDiMEayBMagPrKI7ld-flEqenCIg'
        TEST_IMAGES_URL  = 'https://hxppaq.bl.files.1drv.com/y4mM4VFu53lo1i8OW7HhQlmP5YJANItp3B0Wc8UAD4V84pPmy5arhJdpxpvS-mpk_6Rv9POdJnpqpNnOqJ39DR3FfG5rhMisAztLk-wi7ZCQ0F63N1gZRVkz6NQMZLNamTfo818P6tWficovSKTFASeWmdh_q-lp6Pkly6kPo5KREvqwXaFKZAb40duubnevFntFeIqNx78HhwwDJVWgS-r-A'
        TRAIN_ANNO_URL   = 'https://hxppaq.dm.files.1drv.com/y4mpxYK-sFQ95BwB6-uRsrREZ2lr7tSxfp2gkHisKljpflkRP-lgQHBNX91Bh7Q-1_GKqJ5NMJ3_97AixtUvW875pK4OLj2js2Ga5jiFabQfycYTzG8MaJfWHHFcoA6cK0vrn6M_sqGFqobl4zNFCOXHQ'
        TEST_ANNO_URL    = 'https://hxppaq.dm.files.1drv.com/y4mdENOMPzuyOBGAwht99-BJjAen3ZPzRPJHz5cvxH0qK635N2Nh7E8tE8ZLFO2bXNpaPkXLGRW7RRWLdF0nSXR_OUnLyRb9s2JOykBzCklduYht_uUoN9vL9ZgKJKV-tnt4XKytFYtOJXbObNfSN8C5A' 
        save_train_zip_file  = dataset_path / 'train.zip'
        save_test_zip_file   = dataset_path / 'test.zip'
        save_train_anno_file = dataset_path / 'train.csv'
        save_test_anno_file  = dataset_path / 'test_kaggletest.csv'
        wget.download(TRAIN_IMAGES_URL, out=str(save_train_zip_file))
        wget.download(TEST_IMAGES_URL, out=str(save_test_zip_file))
        wget.download(TRAIN_ANNO_URL, out=str(save_train_anno_file))
        wget.download(TEST_ANNO_URL, out=str(save_test_anno_file))

        with zipfile.ZipFile(save_train_zip_file, 'r') as z_f:
            z_f.extractall(str(dataset_path))
        with zipfile.ZipFile(save_test_zip_file, 'r') as z_f:
            z_f.extractall(str(dataset_path))
        save_train_zip_file.unlink()
        save_test_zip_file.unlink()
        print(f'Dataset have downloaded and extracted')
    elif dataset=='aliproducts':
        DATASET_PARTS = ['https://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/231780/AliProducts/100001585554035/train_val.part1.tar.gz',
                         'https://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/231780/AliProducts/200001585540031/train_val.part2.tar.gz',
                         'https://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/231780/AliProducts/300001585559032/train_val.part3.tar.gz',
                         'https://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/231780/AliProducts/400001585578035/train_val.part4.tar.gz',
                         'https://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/231780/AliProducts/500001585599038/train_val.part5.tar.gz',
                         'https://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/231780/AliProducts/600001585536030/train_val.part6.tar.gz',
                         'https://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/231780/AliProducts/700001585524033/train_val.part7.tar.gz',
                         'https://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/231780/AliProducts/800001585502035/train_val.part8.tar.gz',
                         'https://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/231780/AliProducts/900001585552031/train_val.part9.tar.gz']
        dataset_path = save_path / 'aliproducts'
        dataset_path.mkdir(exist_ok=True)
        for idx, DATASET_PART in enumerate(DATASET_PARTS):
            save_part_file  = dataset_path / f'train_val.part{idx+1}.tar.gz'
            wget.download(DATASET_PART, out=str(save_part_file))
            with tarfile.open(save_part_file) as tar:
                tar.extractall(dataset_path)
            save_part_file.unlink()
        print(f'Dataset have downloaded and extracted')
    elif dataset=='h_and_m':
        dataset_path = save_path / 'h_and_m'
        dataset_path.mkdir(exist_ok=True)

        try:
            print(f'Downloading dataset...')
            os.system(f"kaggle competitions download -c h-and-m-personalized-fashion-recommendations -p {str(dataset_path)}")

            z_file = dataset_path / 'h-and-m-personalized-fashion-recommendations.zip'
            with zipfile.ZipFile(z_file, 'r') as z_f:
                z_f.extractall(str(dataset_path))

            (dataset_path / 'transactions_train.csv').unlink()
            (dataset_path / 'sample_submission.csv').unlink()
            z_file.unlink()
            print(f'Dataset have downloaded and extracted')
        except:
            print('In order to download H&M dataset you must have kaggle account and generated kaggle.json file in ~/.kaggle/kaggle.json ')
    elif dataset=='rp2k':
        dataset_path = save_path / 'rp2k'
        dataset_path.mkdir(exist_ok=True)

        DATASET_URL = 'https://blob-nips2020-rp2k-dataset.obs.cn-east-3.myhuaweicloud.com/rp2k_dataset.zip'
        z_file  = dataset_path / 'rp2k_dataset.zip'

        wget.download(DATASET_URL, out=str(z_file))

        with zipfile.ZipFile(z_file, 'r') as z_f:
            z_f.extractall(str(dataset_path))
        z_file.unlink()

        print(f'Dataset have downloaded and extracted')
    elif dataset=='largefinefoodai':
        dataset_path = save_path / 'largefinefoodai'
        dataset_path.mkdir(exist_ok=True)

        TRAIN_URL = 'https://s3plus.meituan.net/v1/mss_fad1a48f61e8451b8172ba5abfdbbee5/foodai-workshop-challenge/Train.tar'
        VAL_URL   = 'https://s3plus.meituan.net/v1/mss_fad1a48f61e8451b8172ba5abfdbbee5/foodai-workshop-challenge/Val.tar'

        train_file  = dataset_path / 'Train.tar'
        val_file    = dataset_path / 'Val.tar'

        wget.download(TRAIN_URL, out=str(train_file))
        wget.download(VAL_URL, out=str(val_file))

        with tarfile.open(train_file) as tar:
            tar.extractall(dataset_path)
        train_file.unlink()

        with tarfile.open(val_file) as tar:
            tar.extractall(dataset_path)
        val_file.unlink()
        print(f'Dataset have downloaded and extracted')
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


def get_labels_and_paths(path, split=''):
    class_folders_paths = sorted(list(path.glob('*/')))

    labels = []
    image_names = []

    for class_folder_path in class_folders_paths:
        images = sorted([l for l in list(class_folder_path.glob('*.jpeg')) + \
                                    list(class_folder_path.glob('*.jpg')) + \
                                    list(class_folder_path.glob('*.png'))])
        label = class_folder_path.name
        for img in images:
            labels.append(label)
            if split: 
                image_names.append(f'{split}/{label}/{img.name}')
            else:
                image_names.append(f'{label}/{img.name}')

    return pd.DataFrame(list(zip(image_names, labels)), 
                        columns =['file_name', 'label'])