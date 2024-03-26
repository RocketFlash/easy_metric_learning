import sys
sys.path.append("./")

from src.utils import (plot_embeddings, 
                       plot_embeddings_interactive)
from src.data.utils import get_labels_to_ids
import argparse
import os
from pathlib import Path
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', type=str, default='', help='path to embeddings')
    parser.add_argument('--method', type=str, default='fast_tsne', help='embeddings projection method [fast_tsne, umap, tsne]')
    parser.add_argument('--n_labels', type=int, default=-1, help='number of labels to show, if value is less than 0 for example -5 script will visualize only labels with 5 or more samples')
    parser.add_argument('--n_jobs', type=int, default=4, help='number of parallel jobs')
    parser.add_argument('--interactive', action='store_true', help='use interactive visualization')
    parser.add_argument('--plot_3d', action='store_true', help='save 3d visualization')
    parser.add_argument('--mapper', type=str, default='', help='file path to json mapper from ids to labels')
    parser.add_argument('--save_path', default="./results", help='tmp')
    parser.add_argument('--save_name', default=None, help='save_name')
    parser.add_argument('--show_images', action='store_true', help='show images on dots')
    parser.add_argument('--dataset_path', default="", help='tmp')
    args = parser.parse_args()

    mapper = None
    if args.mapper:
        assert os.path.isfile(args.mapper)
        mapper=get_labels_to_ids(args.mapper)

    save_path = Path(args.save_path)
    save_path.mkdir(exist_ok=True, parents=True)

    print('Load embeddings')
    
    data = np.load(args.embeddings,
                   allow_pickle=True)
    embeddings = data['embeddings']
    labels = data['labels']
    file_names = data['file_names']
    print('Embeddings were loaded')

    dataset_path = args.dataset_path

    if args.interactive:
        n_components = 3 if args.plot_3d else 2
        plot_embeddings_interactive(
            embeddings, 
            labels,
            file_names, 
            save_path, 
            dataset_path=dataset_path,
            n_labels=args.n_labels, 
            mapper=mapper, 
            n_jobs=args.n_jobs,
            method=args.method,
            save_name=args.save_name, 
            n_components=n_components
        )
    else:
        plot_embeddings(
            embeddings, 
            labels, 
            save_path, 
            n_labels=args.n_labels, 
            mapper=mapper, 
            method=args.method,
            n_jobs=args.n_jobs
        )
