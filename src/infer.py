import numpy as np
import cv2
import time
from .transforms import get_transformations
import torch
from tqdm import tqdm

def generate_embeddings(embeddings_model, 
                       dataloader,  
                       device='cpu'):

    data_labels, data_encodings = [], []
    encoded_data = {}

    tqdm_bar = tqdm(dataloader, total=int(len(dataloader)))

    embeddings_model.eval()
    with torch.no_grad():
        for batch_index, (data, targets) in enumerate(tqdm_bar):
            data = data.to(device)
            output = embeddings_model(data)
            encodings = output.cpu().numpy()
            for encoding, label in zip(encodings, targets):
                data_encodings.append(encoding)
                data_labels.append(int(label.cpu()))

    encoded_data['labels'] = np.array(data_labels)
    encoded_data['embeddings'] = np.array(data_encodings)
    
    return encoded_data


def get_embeddings(image, embeddings_model, transform=None, device='cpu'):
    if transform:
        sample = transform(image=image)
        image = sample['image']
    image = torch.from_numpy(image)
    img_torch = image.unsqueeze(0)
    images = img_torch.to(device=device)
    embeddings_model.eval()
    output = embeddings_model.predict(images)
    return output.squeeze().cpu().numpy()


def infer(image, embeddings_model,
                 input_size=(400, 400), 
                 show_time=False,
                 device='cpu',
                 config=None):
    
    if config:
        input_size=(config["DATA"]["IMG_SIZE"], 
                    config["DATA"]["IMG_SIZE"])
                               
        device=config["GENERAL"]["DEVICE"]

    transform = get_transformations('test_aug', image_size=input_size)
    t = time.time()

    output = get_embeddings(image, embeddings_model, transform, device)
    processing_time = (time.time() - t)

    if show_time:
        return output, processing_time
    else:
        return output