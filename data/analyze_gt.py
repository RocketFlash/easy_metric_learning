import sys
sys.path.append("../")

import cv2
import numpy as np
import argparse
from pathlib import Path
from src.utils import plot_embeddings, show_images
from tqdm import tqdm

def get_image(image_path):
    image = cv2.imread(str(image_path))
    image = cv2.resize(image, (170, 170))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_embeddings_file', type=str, default='', help='path to embeddings file')
    parser.add_argument('--gt_dataset_path', type=str, default='', help='dataset path')
    parser.add_argument('--embeddings_file', type=str, default='', help='path to embeddings file')
    parser.add_argument('--dataset_path', type=str, default='', help='dataset path')
    parser.add_argument('--save_path', type=str, default='', help='save path')
    args = parser.parse_args()
    save_path = Path(args.save_path)
    save_path.mkdir(exist_ok=True, parents=True)
    dataset_path = Path(args.dataset_path)
    gt_dataset_path = Path(args.gt_dataset_path)

    gt_data = np.load(args.gt_embeddings_file)
    gt_embeddings = gt_data['embeddings']
    gt_labels = gt_data['labels']
    gt_file_names = gt_data['file_names']
    gt_classes = np.array([str(f_n.split('/')[0]) for f_n in gt_file_names])
    gt_all_classes = np.unique(gt_classes)

    data = np.load(args.embeddings_file)
    embeddings = data['embeddings']
    labels = data['labels']
    file_names = data['file_names']
    classes = np.array([str(f_n.split('/')[0]) for f_n in file_names])
    all_classes = np.unique(classes)

    intersect_labels = np.intersect1d(all_classes, gt_all_classes)

    for l in tqdm(intersect_labels):
        l = str(l)

        embeddings_i = embeddings[classes==l]
        f_names_i = file_names[classes==l]
        if f_names_i.shape[0]>0:
            images_i = [get_image(dataset_path / f_name) for f_name in f_names_i]

            gt_embeddings_i = gt_embeddings[gt_classes==l]
            gt_f_names_i = gt_file_names[gt_classes==l]
            gt_images_i = [get_image(gt_dataset_path / f_name) for f_name in gt_f_names_i]

            embgs = np.concatenate([embeddings_i, gt_embeddings_i]) 
            images = images_i + gt_images_i
            lbls = [1] * len(images_i) + [0] * len(gt_images_i)

            show_images(images_i, save_name=save_path/f'{l}_images.png')
            show_images(gt_images_i, save_name=save_path/f'{l}_images_gt.png')
            plot_embeddings(embgs, 
                            lbls,
                            method='tsne',
                            save_path=save_path/f'{l}_tsne.png')