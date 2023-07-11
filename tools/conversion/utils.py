import torch
from torch.profiler import profile, record_function, ProfilerActivity
import pprint
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path


def generate_representative_dataset(images_path,
                                    image_size=(224, 224),
                                    n_max=200,
                                    save_path='./weights/'):
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


def test_model(model, sample, to_profile=False):
    if to_profile:
        profile_model(model, sample)
    with torch.no_grad():
        o_t_o = model(sample)
        pprint.pprint(o_t_o[:, :8])
        