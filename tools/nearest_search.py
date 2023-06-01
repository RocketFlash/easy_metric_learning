import sys
sys.path.append("./")

import argparse
import os
from pathlib import Path
import numpy as np
from src.utils import cosine_similarity_chunks
from multiprocessing import Pool
from multiprocessing import cpu_count
import pandas as pd
import faiss
import torch
import time
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='search ')
    parser.add_argument('--embeddings', type=str, default='', help='path to embeddings')
    parser.add_argument('--ref_csv', type=str, default='', help='path to references csv')
    parser.add_argument('--test_csv', type=str, default='', help='path to test csv')
    parser.add_argument('--top_n', type=int, default=5, help='number of nearest neighbors')
    parser.add_argument('--n_jobs', type=int, default=4, help='number of parallel jobs')
    parser.add_argument('--n_chunks', type=int, default=100, help='number of chunks')
    parser.add_argument('--filter_labels', action='store_true', help='test samples only on labels presented in test set')
    parser.add_argument('--fold', type=int, default=-1, help='tmp')
    parser.add_argument('--save_path', default="", help='tmp')
    parser.add_argument('--save_name', default="nearest", help='tmp')
    parser.add_argument('--no_faiss', action='store_true', help='Do not use faiss')
    return parser.parse_args()


def get_embeddings_dict(embeddings_path):
    print('Load embeddings...')
    data = np.load(embeddings_path, allow_pickle=True)
    embeddings = data['embeddings']
    labels = data['labels']
    file_names = data['file_names']
    print('Embeddings were loaded')
    fname_to_embeddings = {}
    for f_n, l_n, em in zip(file_names, labels, embeddings):
        fname_to_embeddings[f_n] = (l_n, em)
    return fname_to_embeddings


def filter_embeddings(fname_to_embeddings, df):
    embeddings = []
    labels = []
    f_names = []
    for smpl in df.file_name:
        if smpl in fname_to_embeddings:
            lbl, em = fname_to_embeddings[smpl]
            embeddings.append(em)
            labels.append(lbl)
            f_names.append(smpl)
    return np.array(embeddings), np.array(labels), np.array(f_names)


if __name__ == '__main__':
    args = parse_args()

    if not args.save_path:
        args.save_path = Path(args.embeddings).parents[0]

    save_path = Path(args.save_path)
    save_path.mkdir(exist_ok=True, 
                    parents=True)

    embeddings_dict = get_embeddings_dict(args.embeddings)

    df = pd.read_csv(args.ref_csv, dtype={'label': str,
                                          'file_name': str,
                                          'width': int,
                                          'height': int})

    if args.fold<0:
        train_df = df
        test_df  = pd.read_csv(args.test_csv, dtype={'label': str,
                                                     'file_name': str,
                                                     'width': int,
                                                     'height': int})
    else:
        train_df = df[(df.fold != args.fold)]
        test_df  = df[(df.fold == args.fold)]

    print('N train samples', len(train_df))
    print('N test samples', len(test_df))
    train_embeddings, train_labels, train_fnames = filter_embeddings(embeddings_dict, train_df)
    test_embeddings, test_labels, test_fnames    = filter_embeddings(embeddings_dict, test_df)

    print('N train embeddings', train_embeddings.shape)
    print('N test  embeddings', test_embeddings.shape)

    if args.filter_labels:
        train_labels_set = list(set(train_labels))
        test_embeddings  = test_embeddings[np.isin(test_labels, train_labels_set)]
        test_labels = test_labels[np.isin(test_labels, train_labels_set)]

    print(f'Number of train samples: {len(train_embeddings)}')
    print(f'Number of test  samples: {len(test_embeddings)}')

    print(f'Number of train labels: {len(set(train_labels))}')
    print(f'Number of test  labels: {len(set(test_labels))}')

    if not args.no_faiss:
        print('Calculate similarity using FAISS')
        start = time.time()
        vector_dimension = train_embeddings.shape[1]
        faiss.omp_set_num_threads(args.n_jobs)

        index = faiss.IndexFlatIP(vector_dimension)
        if torch.cuda.is_available():
            print('Use FAISS GPU')
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        faiss.normalize_L2(train_embeddings)
        index.add(train_embeddings)
        faiss.normalize_L2(test_embeddings)
        distances, best_top_n_idxs = index.search(test_embeddings, k=args.top_n)
        end = time.time()
        print(f'FAISS search time: {end - start} seconds')
    else:
        best_top_n_vals, best_top_n_idxs = cosine_similarity_chunks(train_embeddings, 
                                                                    test_embeddings, 
                                                                    n_chunks=args.n_chunks, 
                                                                    top_n=args.top_n)
        best_top_n_idxs = best_top_n_idxs.T
        distances = best_top_n_vals.T

    all_pred = []
    all_gts  = []
    all_fnms = []
    all_dist = []

    for i in tqdm(range(len(best_top_n_idxs))):
        btni       = best_top_n_idxs[i]
        btnd       = distances[i]
        test_label = test_labels[i]
        test_fname = test_fnames[i]
        predicts   = train_labels[btni]

        all_pred.append(predicts)
        all_gts.append(test_label)
        all_fnms.append(test_fname)
        all_dist.append(btnd)

    df = pd.DataFrame(list(zip(all_fnms, all_gts, all_pred, all_dist)), 
                      columns =['file_name', 'gt', 'prediction', 'similarity'])
    start = time.time()
    df.to_feather(save_path / f'{args.save_name}_top{args.top_n}.feather')
    end = time.time()
    print(f'Dataframe save time: {end - start} seconds')

    
