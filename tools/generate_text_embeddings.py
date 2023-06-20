import pandas as pd
from pathlib import Path
import json
import argparse
import numpy as np
from tqdm import tqdm
import open_clip
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='analyze dataset descriptions')
    parser.add_argument('--model_name', default="openclip-ViT-B32_laion2b", help="model name")
    parser.add_argument('--dataset_csv', default="", help="path to the dataset info csv file")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    dataset_csv = Path(args.dataset_csv)
    df = pd.read_csv(dataset_csv, dtype={'label': str,
                                              'file_name': str,
                                              'width': int,
                                              'height': int,
                                              'hash' : str,
                                              'name' : str,
                                              'category' : str})

    if args.model_name == 'openclip-ViT-H_laion2b':
        m_name, p_name = 'ViT-H-14', 'laion2b_s32b_b79k'
    elif args.model_name == 'openclip-ViT-L_laion2b':
        m_name, p_name = 'ViT-L-14', 'laion2b_s32b_b82k'
    elif args.model_name == 'openclip-ViT-B32_laion2b':
        m_name, p_name = 'ViT-B-32', 'laion2b_s34b_b79k'
    elif args.model_name == 'openclip-ViT-B16_laion2b':
        m_name, p_name = 'ViT-B-16', 'laion2b_s34b_b88k'
    elif args.model_name == 'openclip-ConvNext-Base':
        m_name, p_name = 'convnext_base_w', 'laion2b_s13b_b82k_augreg'

    clip_model, _, _ = open_clip.create_model_and_transforms(m_name, 
                                                             pretrained=p_name)
    clip_model = clip_model.cuda()
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    text_embebddings = []
    df.name = df.name.fillna('')
    df.category = df.category.fillna('')
    bs = 512
    batch = []

    for index, row in tqdm(df.iterrows(), total=len(df)):
        product_name     = row['name']
        product_category = row['category']
        product_name     = product_name.replace('##', ' ')
        product_category = product_category.replace('##', ' ')

        if not product_name and not product_category:
            print('No product name and no category')
        
        text = f' Name: {product_name} Category: {product_category}'

        batch.append(text)

        if len(batch)==bs or index==len(df)-1:
            token_value = tokenizer(batch)
            token_value = token_value.cuda()
            
            with torch.no_grad(), torch.cuda.amp.autocast():
                text_embeddings = clip_model.encode_text(token_value)
                text_embebddings.append(text_embeddings.cpu().numpy())
            batch = []

    text_embebddings = np.concatenate(text_embebddings)
    print('Text embeddings shape: ', text_embebddings.shape)
    print('N of rows in df: ', len(df))

    df['text_embeddings'] = text_embebddings.tolist()
    print(df.head())
    csv_file_name = dataset_csv.stem
    csv_file_path = dataset_csv.parents[0]
    df.to_feather(csv_file_path / f'{csv_file_name}_text_emb.feather')

        
