import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm

if __name__ == '__main__':
    DATASET_PATH = Path('/home/rauf/datasets/retechlabs/metric_learning/final_test')
    IMAGES_PATH = DATASET_PATH / 'JPEGImages_test'
    ANNOTATION_FILE_PATH = IMAGES_PATH / 'xml_parsed_test.csv'
    PREDICTIONS_FILE_PATH = DATASET_PATH / 'output_ht.csv'
    REF_FILE_PATH = DATASET_PATH / 'ref_upc.txt'
    ARCFACE_FILE_PATH = Path('results/arcface_results.csv')

    ref_upcs = []
    with open(REF_FILE_PATH) as file_f:
        for l_i, line in tqdm(enumerate(file_f), total=97878):
            product_upc = line.rstrip()
            ref_upcs.append(str(product_upc))
    
    ref_upc = list(set(ref_upcs))
    

    SAVE_PATH = DATASET_PATH / 'final_annotations'
    SAVE_PATH.mkdir(exist_ok=True) 

    annotations = pd.read_csv(ANNOTATION_FILE_PATH)[["keyframeName", "upc", "xmin", "xmax", "ymin", "ymax"]]
    predictions = pd.read_csv(PREDICTIONS_FILE_PATH)[["keyframeName", "predicted_upc", "xmin", "xmax", "ymin", "ymax"]]
    arcface_preds = pd.read_csv(ARCFACE_FILE_PATH)
    
    annotations['upc'] = annotations['upc'].astype(str)
    predictions['predicted_upc'] = predictions['predicted_upc'].astype(str)
    arcface_preds['prediction'] = arcface_preds['prediction'].astype(str)

    annotations = annotations.drop_duplicates()
    predictions = predictions.drop_duplicates()

    annotations['upc'] = annotations['upc'].apply(lambda x: x.zfill(12))
    predictions['predicted_upc'] = predictions['predicted_upc'].apply(lambda x: x.zfill(12))
    arcface_preds['prediction'] = arcface_preds['prediction'].apply(lambda x: x.zfill(12))

    print(f'annotations len {len(annotations)}')
    print(f'predictions len {len(predictions)}')
    print(f'number of ref labels {len(ref_upc)}')

    annotations_full = pd.merge(annotations, predictions, on=['keyframeName', 'xmin', 'xmax', 'ymin', 'ymax'], how='inner')
    annotations_full = annotations_full.loc[annotations_full['upc'].isin(ref_upcs)].reset_index(drop=True)
    annotations_full['arcface_predictions'] = arcface_preds['prediction']
    print(f'annotations full len {len(annotations_full)}')

    annotations_full = annotations_full.rename(columns={'predicted_upc': 'facenet_predictions'})


    accuracy_calculator_facenet = annotations_full['upc'].eq(annotations_full['facenet_predictions']).value_counts()
    accuracy_calculator_arcface = annotations_full['upc'].eq(annotations_full['arcface_predictions']).value_counts()
    
    pos, neg = accuracy_calculator_facenet.values
    facenet_accuracy = (pos / (pos+neg))*100

    pos, neg = accuracy_calculator_arcface.values
    arcface_accuracy = (pos / (pos+neg))*100

    print(annotations_full)
    print(f'Facenet Accuracy : {facenet_accuracy}')
    print(f'Arcface Accuracy : {arcface_accuracy}')

    annotations_full.to_csv('results/full_comparison.csv', index=False)


