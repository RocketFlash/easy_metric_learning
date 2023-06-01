import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='prepare dataset for testing using csv')
    parser.add_argument('--dataset_csv', default="./", help="path to the csv file describing dataset")
    parser.add_argument('--n', default=3, type=int, help="number of reference images")
    parser.add_argument('--random_seed', default=28, help='random seed')
    parser.add_argument('--save_name', default="ref", help="name of saved files")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    DATASET_CSV = Path(args.dataset_csv)
    DATASET_PATH = DATASET_CSV.parents[0]
    RANDOM_STATE = args.random_seed
    N_REF = args.n

    df = pd.read_csv(DATASET_CSV, dtype={
                                         'label': str,
                                         'file_name': str,
                                         'width': int,
                                         'height': int
                                        })

    counts = df['label'].value_counts()
    counts_df = pd.DataFrame({
                              'label':counts.index, 
                              'frequency':counts.values
                             })
    print(counts_df.describe())

    df_groups = df.groupby('label') 
    dfs_ref = []
    dfs_test = []

    for lbl, gr in tqdm(df_groups):
        dfs_ref.append(gr[:N_REF])
        dfs_test.append(gr[N_REF:])
    
    df_ref = pd.concat(dfs_ref)
    df_test = pd.concat(dfs_test)

    print('N labels ref :', len(df_ref.label.unique()))
    print('N labels test:', len(df_test.label.unique()))
    print('N ref images :', len(df_ref))
    print('N test images:', len(df_test))

    df_ref.to_csv(DATASET_PATH / f'{args.save_name}.csv', index=False)
    df_test.to_csv(DATASET_PATH / f'{args.save_name}_test.csv', index=False)
    

    