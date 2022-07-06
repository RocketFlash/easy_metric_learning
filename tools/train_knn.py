from src.utils import load_embedings, get_mapper, load_embedings_separate
import argparse
import os
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
import joblib
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_embeddings', type=str, default='', help='path to embeddings')
    parser.add_argument('--test_embeddings', type=str, default='', help='path to embeddings')
    parser.add_argument('--k', type=int, default=1, help='number of nearest neighbors')
    parser.add_argument('--labels', type=str, default='', help='path to labels')
    parser.add_argument('--n_jobs', type=int, default=4, help='number of parallel jobs')
    parser.add_argument('--mapper', type=str, default='', help='file path to json mapper from ids to labels')
    parser.add_argument('--filter_labels', action='store_true', help='test samples only on labels presented in test tes')
    parser.add_argument('--save_path', default="./results", help='tmp')
    args = parser.parse_args()

    mapper = None
    if args.mapper:
        assert os.path.isfile(args.mapper)
        mapper=get_mapper(args.mapper)

    save_path = Path(args.save_path)
    save_path.mkdir(exist_ok=True, parents=True)

    print('Load embeddings')
    if args.labels:
        train_embeddings, train_labels = load_embedings_separate(args.train_embeddings, args.labels)
    else:
        train_embeddings, train_labels = load_embedings(args.train_embeddings)
    print('Embeddings were loaded')

    print(f'Number of train samples: {len(train_embeddings)}')
    print(f'Number of train labels: {len(set(train_labels))}')

    print('Start KNN training')
    knn = KNeighborsClassifier(n_neighbors=args.k, n_jobs=-1, weights='distance')
    knn.fit(train_embeddings, train_labels)
    print('KNN training finished')
    print(f'KNN N Classes: {knn.classes_}')
    print(f'KNN samples fitted: {knn.n_samples_fit_}')

    filename = save_path / 'knn_model.sav'
    joblib.dump(knn, filename)

    print('Model saved')

    if args.test_embeddings:
        test_embeddings, test_labels = load_embedings(args.test_embeddings)
        
        if args.filter_labels:
            train_labels_set = list(set(train_labels))
            test_embeddings  = test_embeddings[np.isin(test_labels, train_labels_set)]
            test_labels = test_labels[np.isin(test_labels, train_labels_set)]
        
        print(f'Number of test samples: {len(train_embeddings)}')
        print(f'Number of test labels: {len(set(train_labels))}')

        accuracy = knn.score(test_embeddings, test_labels)

        print(f'Current model accuracy: {accuracy}')

    
