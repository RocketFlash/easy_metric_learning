import torch
import copy
import cv2
from tqdm import tqdm

from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, '../../')
from src.transform import get_transform


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
    

def get_quantization(model,
                     save_path,
                     image_size=(170, 170),
                     images_path='',
                     device='cpu',
                     model_name='model'):
    
    backend = "qnnpack" #qnnpack or fbgemm
    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True)
    model_q_path = save_path / f'{model_name}.pt'

    n_max = 10
    if len(images_path)>n_max:
        images_path = images_path[:n_max]

    model_q = copy.deepcopy(model)
    model_q.eval()

    model_q.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.quantization.prepare(model_q, inplace=True)
    
    
    if not model_q_path.is_file():
        transform = get_transform('test_aug', 
                                  image_size=image_size)

        print('Model calibration')
        with torch.no_grad():
            for image_path in tqdm(images_path):
                image = cv2.imread(str(image_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                augmented = transform(image=image)
                img_torch = augmented['image']
                img_torch = img_torch.unsqueeze(0)

                sample = img_torch.to(device=device)
                model_q(sample)

        torch.quantization.convert(model_q, inplace=True)
        torch.save(model_q.state_dict(), model_q_path)
    else:
        torch.quantization.convert(model_q, inplace=True)
        checkpoint = torch.load(model_q_path)
        model_q.load_state_dict(checkpoint)
        
    model_q.eval()

    return model_q