import sys
sys.path.append("../")

import cv2
import numpy as np
import argparse
from pathlib import Path
from src.utils import plot_embeddings, show_images


def get_image(image_path):
    image = cv2.imread(str(image_path))
    image = cv2.resize(image, (170, 170))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings_file', type=str, default='', help='path to embeddings file')
    parser.add_argument('--dataset_path', type=str, default='', help='dataset path')
    parser.add_argument('--save_path', type=str, default='', help='save path')
    args = parser.parse_args()
    save_path = Path(args.save_path)
    dataset_path = Path(args.dataset_path)

    data = np.load(args.embeddings_file)
    embeddings = data['embeddings']
    labels = data['labels']
    file_names = data['file_names']

    all_labels = np.unique(labels)
    print(len(all_labels))

    for l in all_labels:
        l=0
        l_embeddings = embeddings[labels==l]
        l_f_names = file_names[labels==l]
        l_labels = labels[labels==l]
        print(l_f_names)
        images = [get_image(dataset_path / l_f_name) for l_f_name in l_f_names]
        show_images(images, save_name=save_path/'images.png')
        plot_embeddings(l_embeddings, 
                        l_labels,
                        method='tsne',
                        save_dir=save_path)
        break