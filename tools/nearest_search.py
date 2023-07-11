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
    parser.add_argument('--use_test', action='store_true', help='use is_test column')
    parser.add_argument('--save_path', default="", help='tmp')
    parser.add_argument('--save_name', default="nearest", help='tmp')
    parser.add_argument('--no_faiss', action='store_true', help='Do not use faiss')
    parser.add_argument('--faiss_gpu', action='store_true', help='use faiss gpu')
    parser.add_argument('--verbose', action='store_true', help='print logs')
    return parser.parse_args()


def read_embeddings(embeddings_path):
    data = np.load(embeddings_path, allow_pickle=True)
    embeddings = data['embeddings']
    labels = data['labels']
    file_names = data['file_names']
    eval_status = data['eval_status']
    return embeddings, labels, file_names, eval_status


def get_embeddings_dict(embeddings, labels, file_names):
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

    embeddings, labels, file_names, eval_status = read_embeddings(args.embeddings)
    df, train_df, test_df = None, None, None

    if args.ref_csv:
        df = pd.read_csv(args.ref_csv, dtype={'label': str,
                                              'file_name': str,
                                              'width': int,
                                              'height': int})
        
    if args.use_test and df is not None:
        train_df = df[(df.is_test != 1)]
        test_df  = df[(df.is_test == 1)]
    elif args.test_csv and df is not None:
        if args.fold<0:
            train_df = df
            test_df  = pd.read_csv(args.test_csv, dtype={'label': str,
                                                         'file_name': str,
                                                         'width': int,
                                                         'height': int})
        else:
            train_df = df[(df.fold != args.fold)]
            test_df  = df[(df.fold == args.fold)]

    is_inner = False
    if train_df is not None:
        embeddings_dict = get_embeddings_dict(embeddings, labels, file_names)
        train_embeddings, train_labels, train_fnames = filter_embeddings(embeddings_dict, train_df)
        test_embeddings, test_labels, test_fnames    = filter_embeddings(embeddings_dict, test_df)
    else:
        if eval_status.shape:
            if args.verbose: print('Split on gallery and query')
            mask_gallery = eval_status == 'gallery'
            mask_query   = eval_status == 'query'
            train_embeddings, train_labels, train_fnames = embeddings[mask_gallery], labels[mask_gallery], file_names[mask_gallery]
            test_embeddings, test_labels, test_fnames    = embeddings[mask_query], labels[mask_query], file_names[mask_query]
        else:
            if args.verbose: print('Inner embeddings similarity')
            is_inner = True
            train_embeddings, train_labels, train_fnames = embeddings, labels, file_names
            test_embeddings, test_labels, test_fnames    = embeddings, labels, file_names

    if args.verbose:
        print('N gallery embeddings', train_embeddings.shape)
        print('N query   embeddings', test_embeddings.shape)

    if args.filter_labels:
        train_labels_set = list(set(train_labels))
        test_embeddings  = test_embeddings[np.isin(test_labels, train_labels_set)]
        test_labels = test_labels[np.isin(test_labels, train_labels_set)]

    if args.verbose:
        print(f'Number of gallery samples: {len(train_embeddings)}')
        print(f'Number of query   samples: {len(test_embeddings)}')

        print(f'Number of gallery labels: {len(set(train_labels))}')
        print(f'Number of query   labels: {len(set(test_labels))}')

    K = args.top_n 
    if is_inner:
        K += 1

    if not args.no_faiss:
        if args.verbose: print('Calculate similarity using FAISS')
        start = time.time()
        vector_dimension = train_embeddings.shape[1]
        faiss.omp_set_num_threads(args.n_jobs)

        index = faiss.IndexFlatIP(vector_dimension)
        if args.faiss_gpu:
            if args.verbose: print('Use FAISS GPU')
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        faiss.normalize_L2(train_embeddings)
        index.add(train_embeddings)
        faiss.normalize_L2(test_embeddings)
        distances, best_top_n_idxs = index.search(test_embeddings, k=K)
        end = time.time()
        if args.verbose: print(f'FAISS search time: {end - start} seconds')
    else:
        best_top_n_vals, best_top_n_idxs = cosine_similarity_chunks(train_embeddings, 
                                                                    test_embeddings, 
                                                                    n_chunks=args.n_chunks, 
                                                                    top_n=args.top_n)
        best_top_n_idxs = best_top_n_idxs.T
        distances = best_top_n_vals.T

    if is_inner:
        best_top_n_idxs = best_top_n_idxs[:, 1:]
        distances = distances[:, 1:]
        
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
    if args.verbose: print(f'Dataframe save time: {end - start} seconds')

    
