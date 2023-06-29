import argparse
import pandas as pd
import json
from pathlib import Path
from utils import add_image_sizes, download_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', 
                        type=str, 
                        default='./', 
                        help='dataset path')
    parser.add_argument('--save_path', 
                        type=str, 
                        default='./', 
                        help='save path')
    parser.add_argument('--download', 
                        action='store_true', 
                        help='Dowload images')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.download:
        save_path = Path(args.save_path)
        save_path.mkdir(exist_ok=True)
        download_dataset(save_path, dataset='met')
        dataset_path = save_path / 'met'
    else:
        dataset_path = Path(args.dataset_path)
    save_path = dataset_path

    with open(dataset_path / 'testset.json') as f:
        test_anno = json.load(f)
    with open(dataset_path / 'valset.json') as f:
        val_anno  = json.load(f)

    test_labels = [[anno['path'], anno['MET_id']] for anno in test_anno if 'MET_id' in anno]
    val_labels  = [[anno['path'], anno['MET_id']] for anno in val_anno if 'MET_id' in anno]

    df_test = pd.DataFrame(test_labels, 
                           columns =['file_name', 'label'])
    df_val  = pd.DataFrame(val_labels, 
                           columns =['file_name', 'label'])
    df_test['is_test'] = 1
    df_val['is_test']  = 0

    df_test = pd.concat([df_val, df_test],
                        ignore_index=True)
    class_folders_path = sorted(list(dataset_path.glob('MET/*/')))

    labels = []
    image_names = []

    for class_folder_path in class_folders_path:
        images = sorted([l for l in list(class_folder_path.glob('*.jpeg')) + \
                                    list(class_folder_path.glob('*.jpg')) + \
                                    list(class_folder_path.glob('*.png'))])
        label = class_folder_path.name
        for img in images:
            labels.append(label)    
            image_names.append(f'{label}/{img.name}')

    df_info = pd.DataFrame(list(zip(image_names, labels)), 
                           columns =['file_name', 'label'])
    df_info['file_name'] = df_info['file_name'].apply(lambda x: f'MET/{x}')
    df_info['is_test'] = 0

    
    df_info = pd.concat([df_info, df_test],
                        ignore_index=True)

    df_info['label'] = df_info['label'].apply(lambda x: f'met_{x}')
    
    df_info = add_image_sizes(df_info, 
                              dataset_path)
    
    df_info = df_info[['file_name', 
                       'label', 
                       'width', 
                       'height',
                       'is_test']]

    df_info.to_csv(save_path / 'dataset_info.csv', index=False) 

    