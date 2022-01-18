import os
import argparse
import numpy as np
from pathlib import Path

from src.utils import get_train_val_split, load_ckp, load_config
from src.dataloader import get_loader
from src.model import get_model_embeddings
from src.transforms import get_transformations

import pandas as pd
import torch
import cv2
from tqdm import tqdm

def generate_embeddings_from_file(embeddings_model,
                                  images_folder_path,
                                  csv_file_path,
                                  transform=None,
                                  device='cpu'):

    data_labels, data_encodings = [], []
    encoded_data = {}

    data_info = pd.read_csv(csv_file_path)

    embeddings_model.eval()
    curr_file_name = ''
    with torch.no_grad():
        for index, row in tqdm(data_info.iterrows(), total=len(data_info)):
            file_name = row['keyframeName']
            image_file_path = Path(images_folder_path) / file_name
            x1, x2, y1, y2 = row['xmin'], row['xmax'], row['ymin'], row['ymax']
            label = row['upc']
            if curr_file_name != file_name:
                curr_file_name = file_name
                image = cv2.imread(str(image_file_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bbox_crop = image[y1:y2, x1:x2]

            sample = transform(image=bbox_crop)
            bbox_crop = sample['image']

            bbox_crop = bbox_crop.unsqueeze(0)
            bbox_crop = bbox_crop.to(device=device)

            output = embeddings_model(bbox_crop)
            encoding = output.squeeze().cpu().numpy()

            data_encodings.append(encoding)
            data_labels.append(label)

    encoded_data['labels'] = np.array(data_labels)
    encoded_data['embeddings'] = np.array(data_encodings)
    
    return encoded_data


def main(CONFIGS, images_folder_path, csv_file_path, embeddings_save_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform_test = get_transformations('test_aug', image_size=(CONFIGS['DATA']['IMG_SIZE'],
                                                                 CONFIGS['DATA']['IMG_SIZE']))                        

    model = get_model_embeddings(model_name=CONFIGS['MODEL']['ENCODER_NAME'], 
                                 embeddings_size=CONFIGS['MODEL']['EMBEDDINGS_SIZE'],   
                                 dropout=CONFIGS['TRAIN']['DROPOUT_PROB'])

    model = load_ckp(CONFIGS['TEST']['WEIGHTS'], model, emb_model_only=True)
    model.to(device)

    embeddings = generate_embeddings_from_file(model,
                                               images_folder_path,
                                               csv_file_path,
                                               transform=transform_test, 
                                               device=device)
    
    save_name_full = "embeddings_labels_file.npy"
    np.save(Path(embeddings_save_path)/save_name_full, embeddings)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='', help='path to cfg.yaml')
    parser.add_argument('--images_folder_path', type=str, default='', help='images folder path')
    parser.add_argument('--csv_file_path', type=str, default='', help='path to csv file')
    parser.add_argument('--embeddings_save_path', type=str, default='', help='embeddings save path')
    parser.add_argument('--tmp', default="", help='tmp')
    args = parser.parse_args()

    assert os.path.isfile(args.config)
    CONFIGS = load_config(args.config)

    main(CONFIGS, args.images_folder_path, args.csv_file_path, args.embeddings_save_path)