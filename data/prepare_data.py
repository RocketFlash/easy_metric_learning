import pandas as pd
from pathlib import Path
import json
from utils import get_labels_to_ids_map, get_stratified_kfold


if __name__ == '__main__':
    DATASET_PATH = Path('/home/rauf/datasets/retechlabs/metric_learning/rebotics')
    IMAGES_PATH = DATASET_PATH / 'images'
    LABELS_FILE = DATASET_PATH / 'labels.txt'
    META_FILE = DATASET_PATH / 'meta.json'
    RANDOM_STATE = 28
    
    with open(META_FILE) as file:
        data = json.load(file)
        image_names = data['images']


    labels = []
    with open(LABELS_FILE) as file:
        for line in file:
            labels.append(line.rstrip())

    labels_to_ids, ids_to_labels = get_labels_to_ids_map(labels)
    with open(DATASET_PATH / 'labels_to_ids.json', 'w') as fp:
        json.dump(labels_to_ids, fp)
    
    with open(DATASET_PATH / 'ids_to_labels.json', 'w') as fp:
        json.dump(ids_to_labels, fp)

    
    df = pd.DataFrame(list(zip(image_names, labels)), columns =['file_name', 'label'])
    df['label_id'] = df['label'].map(labels_to_ids)

    df = get_stratified_kfold(df, k=5, random_state=28)
    df.to_csv(DATASET_PATH / 'folds.csv', index=False)

    print(df)
    print(f'len of image names: {len(image_names)}')
    print(f'len of labels: {len(labels)}')