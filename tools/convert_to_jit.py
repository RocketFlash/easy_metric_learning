import os
import torch
from pathlib import Path
import argparse
from src.model import get_model_embeddings
from src.utils import load_ckp, load_config
from src.transforms import get_transformations
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='', help='path to cfg.yaml')
    parser.add_argument('--model', required=True, help='path to the pretrained model')
    parser.add_argument('--device', type=str, default='', help='device')
    parser.add_argument('--tmp', default="./", help='tmp')
    args = parser.parse_args()

    assert os.path.isfile(args.config)
    CONFIGS = load_config(args.config)

    if args.tmp != "" and args.tmp != CONFIGS["MISC"]["TMP"]:
        CONFIGS["MISC"]["TMP"] = Path(args.tmp)
        CONFIGS["MISC"]["TMP"].mkdir(exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    transform_test = get_transformations('test_aug_tiles', image_size=(CONFIGS['DATA']['IMG_SIZE'],
                                                                       CONFIGS['DATA']['IMG_SIZE']))
    
    sample = transform_test(image=np.zeros((1000,1000,3), np.uint8))
    example = sample['image']
    example = example.unsqueeze(0).to(device)

    model = get_model_embeddings(model_name=CONFIGS['MODEL']['ENCODER_NAME'], 
                                 embeddings_size=CONFIGS['MODEL']['EMBEDDINGS_SIZE'],   
                                 dropout=CONFIGS['TRAIN']['DROPOUT_PROB'])
    
    model_path = Path(args.model)
    model = load_ckp(model_path, model, emb_model_only=True)
    model.to(device)
    model.eval()

    model_jit = torch.jit.trace(model, example)
    model_jit = torch.jit.optimize_for_inference(model_jit)
    model_jit.save(Path(args.tmp)/f'{model_path.stem}.pt')