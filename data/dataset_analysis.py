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
    parser.add_argument('--dataset_path', default="./", help="path to the dataset")
    parser.add_argument('--save_path', default="./", help='visualization save directory')

    args = parser.parse_args()

    DATASET_PATH = Path(args.dataset_path)
    CLASSES_FOLDER_PATHS = sorted(list(DATASET_PATH.glob('*/')))
    SAVE_PATH = Path(args.save_path)
    SAVE_PATH.mkdir(exist_ok=True, parents=True)
    
    
    labels = []
    image_names = []
    labels_dict = {}

    for class_folder_path in tqdm(CLASSES_FOLDER_PATHS):
        images = sorted([l for l in list(class_folder_path.glob('*.jpeg')) + \
                           list(class_folder_path.glob('*.jpg')) + \
                           list(class_folder_path.glob('*.png'))])
        label = class_folder_path.name
        for img in images:
            labels.append(label)
            if label not in labels_dict:
                labels_dict[label] = 1
            else:
                labels_dict[label]+=1  

            image_names.append(f'{label}/{img.name}')

    data_tuples = list(zip(image_names, labels))
    df = pd.DataFrame(data_tuples, columns=['filename','label'])


    counts = df.label.value_counts()
    counts_df = pd.DataFrame({'label':counts.index, 'frequency':counts.values})


    get_percentage_of_classes_less_than(counts_df, threshold=[2,5,10,20,50,100,500])

    counts_df = counts_df.head(50)

    counts_df['label'] =   counts_df.label.apply(lambda x: f'{x}')

    fig = px.bar(counts_df, x="frequency", y="label",color='label', orientation='h',
                            hover_data=["label", "frequency"],
                            height=1000,
                            title='Number of images per label (Top 50 labels)')
    

    fig.write_image(SAVE_PATH / 'top50_counts.png')