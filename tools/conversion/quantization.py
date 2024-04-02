import torch
from torch import nn
import torch.quantization as quantization
import copy
import cv2
from tqdm import tqdm

from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, './')
from tools.conversion.utils import get_sample


def generate_representative_dataset(images_path, save_path='./weights/calibdata.npy'):
    n_max = 200
    if len(images_path)>n_max:
        images_path = images_path[:n_max]
        
    image_size = (512, 512)
    
    img_datas = []
    for image_path in tqdm(images_path):
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, image_size)

        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)
        img_datas.append(image)

    calib_datas = np.vstack(img_datas)
    print(f'calib_datas.shape: {calib_datas.shape}')
    np.save(file=save_path, arr=calib_datas)
    

def get_quantized_model(
        model,
        save_path,
        transform,
        image_paths,
        device='cpu',
        model_save_path='model.pt',
        n_images_max=1000,
        q_config='x86',
        dynamic=False
    ):
    
    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True)
    
    if len(image_paths)>n_images_max:
        image_paths = image_paths[:n_images_max]

    model_q = copy.deepcopy(model)
    model_q.eval()

    model_q.qconfig = quantization.get_default_qconfig(q_config)
    torch.backends.quantized.engine = q_config

    if dynamic:
        model_q = quantization.quantize_dynamic(
            model=model_q, 
            qconfig_spec={nn.Linear}, 
            dtype=torch.qint8,
        )
        torch.save(model_q.state_dict(), model_save_path)
    else:
        model_q = nn.Sequential(
            torch.quantization.QuantStub(), 
            model_q, 
            torch.quantization.DeQuantStub()
        )

        quantization.prepare(model_q, inplace=True)
        
        if not model_save_path.is_file():
            print('Model calibration')
            with torch.no_grad():
                for image_path in tqdm(image_paths):
                    image = get_sample(
                        image_path,
                        transform=transform
                    ).to(device=device)
                    model_q(image)

            torch.quantization.convert(model_q, inplace=True)
            torch.save(model_q.state_dict(), model_save_path)
        else:
            torch.quantization.convert(model_q, inplace=True)
            checkpoint = torch.load(model_save_path, map_location=device)
            model_q.load_state_dict(checkpoint)
        
    model_q.eval()

    return model_q