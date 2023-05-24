import sys
sys.path.append("./")

import argparse
import os
from pathlib import Path
import numpy as np
from multiprocessing import Pool
from multiprocessing import cpu_count
import pandas as pd
from tqdm import tqdm
from collections import Counter
import pprint
import json
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import random
import cv2


def get_image(image_path):
    image = cv2.imread(str(image_path))
    image = cv2.resize(image, (170, 170))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

def get_random_images(dataset_path, class_name, n_samples=1):
    class_path = dataset_path / class_name
    images_paths_all =  list(class_path.glob('*.png')) +\
                        list(class_path.glob('*.jpg')) +\
                        list(class_path.glob('*.PNG')) +\
                        list(class_path.glob('*.JPG')) +\
                        list(class_path.glob('*.JPEG')) +\
                        list(class_path.glob('*.jpeg'))

    images_paths = random.sample(images_paths_all, n_samples)
    return [get_image(im_path) for im_path in images_paths]

def generate_images_grid(class_name, similarity_dict, dataset_path, n_samples=1, save_path='./'):
    n_sim = len(similarity_dict)

    fig = plt.figure(figsize=(15, 15))
    grid = ImageGrid(fig, 111,
                 nrows_ncols=(n_samples, n_sim+1),
                 axes_pad=0.2)

    gt_images = get_random_images(dataset_path, class_name, n_samples)

    sim_classes_images = []
    titles_arranged  = [f'GT {class_name}']
    
    for sim_class, sim_score in similarity_dict.items():
        sim_images = get_random_images(dataset_path, sim_class, n_samples)
        sim_classes_images.append(sim_images)
        titles_arranged.append(f'{sim_class} similarity: ({sim_score})')
    

    images_arranged = []
    for i in range(len(gt_images)):
        gt_image = gt_images[i]
        images_arranged.append(gt_image)
        for sim_class_images in sim_classes_images:
            images_arranged.append(sim_class_images[i])

    
    for im_idx, (ax, im) in enumerate(zip(grid, images_arranged)):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([]) 
        if im_idx<len(titles_arranged):
            ax.set_title(titles_arranged[im_idx], fontsize = 14)  

    fig.savefig(save_path / f'{class_name}.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_csv', type=str, default='', help='path to references csv')
    parser.add_argument('--dataset_path', type=str, default='', help='path to references csv')
    parser.add_argument('--save_name', type=str, default='similar_stat', help='name of saved file')
    parser.add_argument('--min_similarity', type=float, default=0.5, help='path to references csv')
    args = parser.parse_args()

    MIN_SIMILARITY = args.min_similarity
    DATASET_PATH =Path(args.dataset_path)
    CSV_PATH = Path(args.result_csv)
    SAVE_PATH = CSV_PATH.parents[0]
    SAVE_FILE_NAME = SAVE_PATH / f'{args.save_name}.json'

    if not SAVE_FILE_NAME.is_file():
        df = pd.read_csv(CSV_PATH, dtype={'file_name': str,
                                            'gt': str,
                                            'prediction': str})

        classes_groups = df.groupby("gt")
        similar_classes_dict = {}
        for class_name, class_group in tqdm(classes_groups):
            for index, row in class_group.iterrows():
                similar_classes = row['prediction'].replace('\'', '')
                similar_classes = similar_classes.replace('\n', '')
                similar_classes = similar_classes.replace('[', '')
                similar_classes = similar_classes.replace(']', '')
                similar_classes = similar_classes.split(' ')
                similar_classes_dict[class_name] = similar_classes_dict.get(class_name, []) + similar_classes

        similar_classes_stat = {}
        for class_name, similar_classes in similar_classes_dict.items():
            n_predictions = len(similar_classes)
            similar_classes = [cl_n for cl_n in similar_classes if cl_n != class_name]
            similar_classes_counter = Counter(similar_classes)
            if similar_classes_counter:
                similar_classes_counter = {k: float(v) /n_predictions for k, v in similar_classes_counter.items()}
                similar_classes_stat[class_name] = similar_classes_counter

        with open(SAVE_FILE_NAME, "w") as outfile:
            json.dump(similar_classes_stat, outfile)
    else:
        with open(SAVE_FILE_NAME, 'r') as openfile:
            similar_classes_stat = json.load(openfile)
    

    connectivity_graph = {}
    bidirectional_connections = {}

    for class_name, class_sim_stat in similar_classes_stat.items():
        class_sim_stat = dict(filter(lambda item: item[1]>=MIN_SIMILARITY, class_sim_stat.items()))
        if class_sim_stat:
            connectivity_graph[class_name] = class_sim_stat


    for class_name, sim_classes in connectivity_graph.items():
        for sim_class, sim_score in sim_classes.items():
            if sim_class in connectivity_graph:
                if class_name in connectivity_graph[sim_class].keys():
                    curr_dict = bidirectional_connections.get(class_name, {})
                    curr_dict[sim_class] = sim_score
                    bidirectional_connections[class_name] = curr_dict

    
    pprint.pprint(bidirectional_connections)
    
    print('Similarity dict len: ', len(similar_classes_stat))
    print('Connectivity graph len: ', len(connectivity_graph))
    print('bidirectional graph len: ', len(bidirectional_connections))


    for class_name, similarity_dict in tqdm(bidirectional_connections.items()):
        generate_images_grid(class_name, similarity_dict, DATASET_PATH, n_samples=2, save_path=SAVE_PATH)
