import imagesize
import numpy as np
import pandas as pd


class BaseDataset:
    def __init__(self, save_path, dataset_folder='./'):
        self.dataset_path = save_path
        self.dataset_folder = dataset_folder


    def download(self):
        pass


    def prepare(self):
        pass


    def add_image_sizes(self, df, images_folder):
        image_sizes = []

        for index, row in df.iterrows():
            image_path = images_folder / row.file_name
            width, height = imagesize.get(image_path)
            image_sizes.append([width, height])

        image_sizes = np.array(image_sizes)
        df['width']  = list(image_sizes[:, 0])
        df['height'] = list(image_sizes[:, 1])
        return df

    
    def get_labels_and_paths(self, path, split=''):
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