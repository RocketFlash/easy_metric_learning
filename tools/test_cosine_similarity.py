from src.utils import load_embedings, get_mapper
import argparse
import os
from pathlib import Path
import numpy as np
from src.utils import cosine_similarity_chunks
from multiprocessing import Pool
from multiprocessing import cpu_count
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_embeddings', type=str, default='', help='path to embeddings')
    parser.add_argument('--test_embeddings', type=str, default='', help='path to embeddings')
    parser.add_argument('--train_mapper', type=str, default='', help='file path to json mapper from ids to labels')
    parser.add_argument('--test_mapper', type=str, default='', help='file path to json mapper from ids to labels')
    parser.add_argument('--top_n', type=int, default=5, help='number of nearest neighbors')
    parser.add_argument('--labels', type=str, default='', help='path to labels')
    parser.add_argument('--n_jobs', type=int, default=4, help='number of parallel jobs')
    parser.add_argument('--n_chunks', type=int, default=100, help='number of chunks')
    parser.add_argument('--filter_labels', action='store_true', help='test samples only on labels presented in test tes')
    parser.add_argument('--save_path', default="./results", help='tmp')
    args = parser.parse_args()

    train_mapper = None
    test_mapper = None
    if args.train_mapper:
        assert os.path.isfile(args.train_mapper)
        train_mapper=get_mapper(args.train_mapper)
    if args.test_mapper:
        assert os.path.isfile(args.test_mapper)
        test_mapper=get_mapper(args.test_mapper)

    save_path = Path(args.save_path)
    save_path.mkdir(exist_ok=True, parents=True)

    print('Load embeddings')
    
    train_embeddings, train_labels = load_embedings(args.train_embeddings)
    test_embeddings, test_labels   = load_embedings(args.test_embeddings)
    all_train_labels = [train_mapper[str(p)].zfill(12) for p in train_labels]

    if args.filter_labels:
        train_labels_set = list(set(train_labels))
        test_embeddings  = test_embeddings[np.isin(test_labels, train_labels_set)]
        test_labels = test_labels[np.isin(test_labels, train_labels_set)]

    print('Embeddings were loaded')

    print(f'Number of train samples: {len(train_embeddings)}')
    print(f'Number of train labels: {len(set(train_labels))}')
    print(f'Number of test samples: {len(test_embeddings)}')
    print(f'Number of test labels: {len(set(test_labels))}')
    
    total_n_test_samples = 0
    correct_predictions = 0
    correct_predictions_top_n = 0
    skiped_samples = 0
    n_all_test_samples = len(test_labels)
    
    all_predictions = []
    all_gts = []
    best_top_n_vals, best_top_n_idxs = cosine_similarity_chunks(train_embeddings, test_embeddings, n_chunks=args.n_chunks, top_n=args.top_n)
    best_top_n_idxs = best_top_n_idxs.T

    for btni, test_label in zip(best_top_n_idxs, test_labels):
        if test_mapper is not None:
            test_label_name = test_mapper[str(test_label)].zfill(12)
        else: 
            test_label_name = str(test_label).zfill(12)

        if test_label_name not in all_train_labels:
            skiped_samples+=1
            continue
        predicts = train_labels[btni]
        predicts_labels = [train_mapper[str(p)].zfill(12) for p in predicts]

        all_predictions.append(predicts_labels[0])
        all_gts.append(test_label_name)
        if test_label_name in predicts_labels:
            correct_predictions+=1
        total_n_test_samples+=1

    df = pd.DataFrame(list(zip(all_gts, all_predictions)), columns =['gt', 'prediction'])
    df.to_csv('results/arcface_results.csv', index=False)

    print(f'Total accuracy: {(correct_predictions/total_n_test_samples)*100} %')
    print(f'Number of skipped samples: {skiped_samples} / {n_all_test_samples}')

    
