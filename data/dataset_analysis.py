import pandas as pd
from pathlib import Path
import json
from utils import get_labels_to_ids_map, get_stratified_kfold
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import plotly.express as px

def get_percentage_of_classes_less_than(counts_df, threshold=100):
    total_n_classes = len(counts_df)
    if isinstance(threshold, list):
        for tr in threshold:
            print(f"Number of classes under {tr:5} occurences: ",((counts_df['frequency'] <= tr).sum()/total_n_classes) * 100, '%')
    else:
        print(f"Number of classes under {threshold:5} occurences: ",((counts_df['frequency'] <= threshold).sum()/total_n_classes) * 100, '%')

if __name__ == '__main__':
    DATASET_PATH = Path('/home/rauf/datasets/retechlabs/metric_learning/sirajul_dataset/train_set/')
    CLASSES_FOLDER_PATHS = sorted(list(DATASET_PATH.glob('*/')))
    SAVE_PATH = Path('../results/dataset_analysis/')
    SAVE_PATH.mkdir(exist_ok=True)
    
    EXT = '.png'
    
    labels = []
    image_names = []
    labels_dict = {}

    for class_folder_path in tqdm(CLASSES_FOLDER_PATHS):
        images = sorted(list(class_folder_path.glob(f'*{EXT}')))
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