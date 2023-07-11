import pandas as pd
import argparse
import numpy as np
import pprint
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='search ')
    parser.add_argument('--nearest_csv', type=str, default='', help='path to nearset labels predictions')
    parser.add_argument('--save_path', type=str, default='', help='results save path')
    return parser.parse_args()


def top_k_accuracy(y_true, y_pred, k=1):
    return np.equal(y_pred[:, :k], y_true[:, None]).any(axis=1).mean()


def get_accuracies(predictions, gts):
    accuracies = {}
    top_ks = predictions.shape[1]
    for k in range(1, top_ks+1):
        accuracies[f'Acc top{k}'] = round(top_k_accuracy(gts, predictions, k=k), 5)
    
    return accuracies


if __name__ == '__main__':
    args = parse_args()

    if not args.save_path:
        args.save_path = Path(args.nearest_csv).parents[0]

    save_path = Path(args.save_path)
    save_path.mkdir(exist_ok=True, 
                    parents=True)

    df = pd.read_feather(args.nearest_csv)

    predictions  = np.array([row.astype(str) for row in df['prediction'].to_numpy()])
    similarities = np.array([row for row in df['similarity'].to_numpy()])
    gts          = df['gt'].to_numpy()
    
    accuracies = get_accuracies(predictions, gts)
    pprint.pprint(accuracies)

    metrics_values = pd.DataFrame(accuracies.items(), columns=['metric', 'score'])
    metrics_values.to_csv(save_path / 'metrics.csv', index=False)
    
