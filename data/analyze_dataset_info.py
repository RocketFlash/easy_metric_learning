import pandas as pd
from pathlib import Path
import argparse
from matplotlib import pyplot as plt
from utils import get_counts_df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_info', 
        type=str, 
        default='./', 
        help='path to dataset info file'
    )
    parser.add_argument(
        '--save_path', 
        type=str, 
        default='tmp', 
        help='plots save path'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    dataset_info_path = Path(args.dataset_info)
    dataset_path = dataset_info_path.parents[0]
    save_path = Path(args.save_path)
    save_path.mkdir(exist_ok=True)

    df = pd.read_csv(
        dataset_info_path, 
        dtype={
            'label': str,
            'file_name': str,
            'width': int,
            'height': int,
            'hash': str
        }
    )

    counts_df = get_counts_df(df)
    counts_df.to_csv(dataset_path / 'class_counts.csv', index=False)
    
    print('N samples: ', len(df))
    print('N labels : ', len(counts_df))
    print(counts_df.describe()) 

    min_frequency = counts_df.frequency.min()
    max_frequency = counts_df.frequency.max()
    bins = list(range(min_frequency, max_frequency+1))
    fig = counts_df.frequency.plot.hist(
        bins=bins, 
        edgecolor='black', 
        linewidth=1.2,
        title='Number of samples per class frequency distribution',
        xlabel='N of samples per class'
    ) 
    plt.savefig(save_path / 'number_of_samples_per_class_frequency_distribution.png')