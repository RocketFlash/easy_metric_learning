import pandas as pd
from pathlib import Path
from tqdm import tqdm
import plotly.express as px
import argparse

def get_percentage_of_classes_less_than(counts_df, threshold=100):
    total_n_classes = len(counts_df)
    if isinstance(threshold, list):
        for tr in threshold:
            print(f"Number of classes under {tr:5} occurences: ",((counts_df['frequency'] <= tr).sum()/total_n_classes) * 100, '%')
    else:
        print(f"Number of classes under {threshold:5} occurences: ",((counts_df['frequency'] <= threshold).sum()/total_n_classes) * 100, '%')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='find dublicates')
    # arguments from command line
    parser.add_argument('--dataset_csv', default="./", help="path to the dataset")
    parser.add_argument('--save_path', default="~/tmp", help='visualization save directory')

    args = parser.parse_args()

    DATASET_CSV = Path(args.dataset_csv)
    DATASET_PATH = DATASET_CSV.parents[0]

    SAVE_PATH = Path(args.save_path)
    SAVE_PATH.mkdir(exist_ok=True, parents=True)
    
    df = pd.read_csv(DATASET_CSV, dtype={'label': str,
                                        'file_name': str,
                                        'width': int,
                                        'height': int,
                                        'hash': str
                                    })
    counts = df['label'].value_counts()
    counts_df = pd.DataFrame({'label':counts.index, 'frequency':counts.values})
    counts_df.to_csv(DATASET_PATH / 'class_counts.csv', index=False)
    print(counts_df.describe()) 

    get_percentage_of_classes_less_than(counts_df, threshold=[2,5,6,7,8,10,20,50,100,500])

    counts_df = counts_df.head(50)

    counts_df['label'] =   counts_df.label.apply(lambda x: f'{x}')

    fig = px.bar(counts_df, x="frequency", y="label",color='label', orientation='h',
                            hover_data=["label", "frequency"],
                            height=1000,
                            title='Number of images per label (Top 50 labels)')
    

    fig.write_image(SAVE_PATH / 'top50_counts.png')