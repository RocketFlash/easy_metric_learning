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

def filter_embeddings(fname_to_embeddings, df):
    embeddings = []
    labels = []
    for smpl in df.file_name:
        if smpl in fname_to_embeddings:
            lbl, em = fname_to_embeddings[smpl]
            embeddings.append(em)
            labels.append(lbl)
    return np.array(embeddings), np.array(labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', type=str, default='', help='path to embeddings')
    parser.add_argument('--ref_csv', type=str, default='', help='path to references csv')
    parser.add_argument('--test_csv', type=str, default='', help='path to test csv')
    parser.add_argument('--top_n', type=int, default=5, help='number of nearest neighbors')
    parser.add_argument('--n_jobs', type=int, default=4, help='number of parallel jobs')
    parser.add_argument('--n_chunks', type=int, default=100, help='number of chunks')
    parser.add_argument('--filter_labels', action='store_true', help='test samples only on labels presented in test set')
    parser.add_argument('--fold', type=int, default=-1, help='tmp')
    parser.add_argument('--save_path', default="./results", help='tmp')
    parser.add_argument('--save_name', default="results", help='tmp')
    parser.add_argument('--faiss', action='store_true', help='Use faiss')
    args = parser.parse_args()

    save_path = Path(args.save_path)
    save_path.mkdir(exist_ok=True, parents=True)

    print('Load embeddings')
    
    data = np.load(args.embeddings)
    embeddings = data['embeddings']
    labels = data['labels']
    file_names = data['file_names']
    print('Embeddings were loaded')

    fname_to_embeddings = {}
    for f_n, l_n, em in zip(file_names, labels, embeddings):
        fname_to_embeddings[f_n] = (l_n, em)

    if args.fold<0:
        train_df = pd.read_csv(args.ref_csv, dtype={'label': str,
                                            'file_name': str,
                                            'width': int,
                                            'height': int,
                                            'label_id': int})
        test_df = pd.read_csv(args.test_csv, dtype={'label': str,
                                            'file_name': str,
                                            'width': int,
                                            'height': int,
                                            'label_id': int})
    else:
        df = pd.read_csv(args.ref_csv, dtype={'label': str,
                                            'file_name': str,
                                            'width': int,
                                            'height': int,
                                            'label_id': int})
        train_df = df[(df.fold != args.fold)]
        test_df = df[(df.fold == args.fold)]

    # train_df = train_df.sample(n = 1000)
    # test_df = test_df.sample(n = 100)
    print('N train samples', len(train_df))
    print('N test samples', len(test_df))

    train_embeddings, train_labels = filter_embeddings(fname_to_embeddings, train_df)
    test_embeddings, test_labels = filter_embeddings(fname_to_embeddings, test_df)

    print('N train embeddings', train_embeddings.shape)
    print('N test  embeddings', test_embeddings.shape)

    if args.filter_labels:
        train_labels_set = list(set(train_labels))
        test_embeddings  = test_embeddings[np.isin(test_labels, train_labels_set)]
        test_labels = test_labels[np.isin(test_labels, train_labels_set)]

    print('Embeddings were loaded')

    print(f'Number of train samples: {len(train_embeddings)}')
    print(f'Number of train labels: {len(set(train_labels))}')
    print(f'Number of test samples: {len(test_embeddings)}')
    print(f'Number of test labels: {len(set(test_labels))}')

    all_predictions = []
    all_gts = []
    total_n_test_samples = 0
    correct_predictions = 0
    correct_predictions_top_n = 0
    n_all_test_samples = len(test_labels)

    if args.faiss:
        vector_dimension = train_embeddings.shape[1]
        faiss.omp_set_num_threads(args.n_jobs)

        index = faiss.IndexFlatIP(vector_dimension)
        if torch.cuda.is_available():
            print('Use FAISS GPU')
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        faiss.normalize_L2(train_embeddings)
        index.add(train_embeddings)
        print('Calculate similarity using FAISS')
        distances, best_top_n_idxs = index.search(test_embeddings, k=args.top_n)
        print(best_top_n_idxs)
    else:
        best_top_n_vals, best_top_n_idxs = cosine_similarity_chunks(train_embeddings, test_embeddings, n_chunks=args.n_chunks, top_n=args.top_n)
        best_top_n_idxs = best_top_n_idxs.T

    for btni, test_label in zip(best_top_n_idxs, test_labels):
        predicts = train_labels[btni]

        all_predictions.append(predicts)
        all_gts.append(test_label)
        if test_label in predicts:
            correct_predictions+=1
        total_n_test_samples+=1

    df = pd.DataFrame(list(zip(all_gts, all_predictions)), columns =['gt', 'prediction'])
    df.to_csv(save_path / f'{args.save_name}_top{args.top_n}.csv', index=False)

    print(f'Total accuracy: {(correct_predictions/total_n_test_samples)*100} %')

    
