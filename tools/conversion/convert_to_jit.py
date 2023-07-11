import sys
sys.path.append("../../")

import os
import torch
from pathlib import Path
import argparse
from src.model import get_model_embeddings
from src.utils import load_ckp, load_config, get_device
from src.transform import get_transform
from torch.quantization import quantize_dynamic
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_folder', type=str, default='', help='path to trained model working directory')
    parser.add_argument('--config', type=str, default='', help='path to cfg.yaml')
    parser.add_argument('--model', required=True, help='path to the trained model')
    parser.add_argument('--device', type=str, default='', help='device')
    parser.add_argument('--save_name', type=str, default='best_emb', help='device')
    parser.add_argument('--mobile', action='store_true', help='optimize for mobile')
    parser.add_argument('--quantize', action='store_true', help='quantize model')
    parser.add_argument('--tmp', default="./", help='tmp')
    args = parser.parse_args()

    assert os.path.isfile(args.config)
    CONFIGS = load_config(args.config)

    if args.tmp != "" and args.tmp != CONFIGS["MISC"]["TMP"]:
        CONFIGS["MISC"]["TMP"] = Path(args.tmp)
        CONFIGS["MISC"]["TMP"].mkdir(exist_ok=True)

    device = get_device(args.device)
    print(f'Device: {device}')

    transform_test = get_transform('test_aug', 
                                    data_type=CONFIGS['DATA']['DATA_TYPE'],
                                    image_size=(CONFIGS['DATA']['IMG_SIZE'],
                                                CONFIGS['DATA']['IMG_SIZE']))
    
    sample = transform_test(image=np.zeros((CONFIGS['DATA']['IMG_SIZE'],
                                            CONFIGS['DATA']['IMG_SIZE'],
                                            3), np.uint8))
    example = sample['image']
    example = example.unsqueeze(0).to(device)

    model = get_model_embeddings(model_config=CONFIGS['MODEL'])
    
    model_path = Path(args.model)
    model = load_ckp(model_path, model, emb_model_only=True, device=device)
    model.to(device)
    model.eval()
    
    if args.quantize:
        model = quantize_dynamic(model, 
                                 {nn.Conv2d, nn.Linear}, 
                                 dtype=torch.qint8)

    model_jit = torch.jit.trace(model, example)
    if args.mobile:
        model_jit = optimize_for_mobile(model_jit)
        model_jit.save(Path(args.tmp)/f'{args.save_name}_mobile.pt')
    else:
        # model_jit = torch.jit.optimize_for_inference(model_jit)
        model_jit.save(Path(args.tmp)/f'{args.save_name}_traced.pt')