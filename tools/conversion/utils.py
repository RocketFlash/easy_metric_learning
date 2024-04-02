import torch
from torch.profiler import (profile, 
                            record_function, 
                            ProfilerActivity)
import pprint
import numpy as np
import cv2
import time
from tqdm import tqdm
from pathlib import Path


def get_images_paths(path):
    pathlib_path = Path(path)
    return [l for l in list(pathlib_path.glob('**/*.jpeg')) + \
                       list(pathlib_path.glob('**/*.jpg')) + \
                       list(pathlib_path.glob('**/*.png'))]


def get_sample(
        image_path, 
        transform,
    ):
    image = read_image(image_path)
    augmented = transform(image=image)
    img_torch = augmented['image']
    return img_torch.unsqueeze(0)


def read_image(image_path):
    image = cv2.imread(str(image_path))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 


def print_model_size(model_name, model_path):
    print(f'{model_name} size: {model_path.stat().st_size/1e6} MB')


def generate_representative_dataset(
        images_path,
        image_size=(224, 224),
        n_max=200,
        save_path='./weights/'
    ):
    save_path = Path(save_path)
    if len(images_path)>n_max:
        images_path = images_path[:n_max]
    
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
    np.save(file=save_path / 'calibdata.npy', arr=calib_datas)


def profile_model(model, sample):
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            with torch.no_grad():
                model(sample)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))


def test_model(
        model, 
        sample, 
        to_profile=False,
        n_times=10,
        model_type='torch'
    ):
    if to_profile:
        profile_model(model, sample)
    with torch.no_grad():
        if model_type=='torch':
            model_output = model(sample)

        execution_time_list = []
        for i in range(n_times):
            s_time = time.time()

            if model_type=='onnx':
                model_output = model.run(
                    None, 
                    {"input": sample.cpu().numpy()}
                )
                model_output = model_output[0]
            else:
                model_output = model(sample)

            execution_time_list.append(time.time()-s_time)
        avg_execution_time = sum(execution_time_list)/len(execution_time_list)
        print(f'Average execution time: {avg_execution_time:.5}')

        pprint.pprint(model_output[:, :8])
        return model_output
        