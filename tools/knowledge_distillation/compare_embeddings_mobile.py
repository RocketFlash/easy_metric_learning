import argparse
import numpy as np
from pathlib import Path
import random
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm import tqdm
import torch
from torch.nn.functional import cosine_similarity
import tensorflow as tf


def get_images_paths(path):
    pathlib_path = Path(path)
    return [l for l in list(pathlib_path.glob('**/*.jpeg')) + \
                       list(pathlib_path.glob('**/*.jpg')) + \
                       list(pathlib_path.glob('**/*.png'))]


def get_sample(image_path, 
               img_h=170, 
               img_w=170):
    image = cv2.imread(image_path)

    transform = A.Compose([
            A.Resize(img_h, img_w),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    augmented = transform(image=image)
    img_torch = augmented['image']
    return img_torch.unsqueeze(0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_s', type=str, default='', help='path to student model')
    parser.add_argument('--model_t', type=str, default='', help='path to teacher model')
    parser.add_argument('--image_path', type=str, default='', help='image path')
    parser.add_argument('--dataset_path', default='', help='path to the dataset')
    parser.add_argument('--n_vals', type=int, default=8, help='number of elements to show')
    parser.add_argument('--n_samples', type=int, default=200, help='number of samples to compare')
    parser.add_argument('--img_size', type=int, default=224, help='input image size')
    parser.add_argument('--device', type=str, default='cpu', help='device type')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    device = torch.device(args.device)
    img_size = int(args.img_size)

    model_t = torch.jit.load(args.model_t)
    model_t.to(device)
    model_t.eval()

    interpreter = tf.lite.Interpreter(model_path=args.model_s)
    model_s = interpreter.get_signature_runner()

    if args.dataset_path:
        dataset_path = Path(args.dataset_path)
        images_paths = get_images_paths(dataset_path)
        random.shuffle(images_paths)
        images_paths = images_paths[:args.n_samples]

        cos_sim_vals = []
        for image_path in tqdm(images_paths):
            sample = get_sample(str(image_path),
                                  img_h=img_size,
                                  img_w=img_size)
            with torch.no_grad():
                o_t = model_t(sample.to(device)).cpu()

                data_np = torch.permute(sample, (0, 2, 3, 1)).numpy()
                tf_data = tf.convert_to_tensor(data_np)
                o_s  = model_s(input=tf_data)['output']
                
                cos_sim = cosine_similarity(torch.from_numpy(o_s), 
                                            o_t)
                cos_sim_vals.append(cos_sim)

        np.set_printoptions(linewidth=200)
        print('student : ', np.round(o_s[:, :args.n_vals], 4))
        print('teacher : ', np.round(o_t[:, :args.n_vals].numpy(), 4))
        print(f'avg cosine similarity: {sum(cos_sim_vals)/len(cos_sim_vals)}')
            

    if args.image_path:
        sample_s = get_sample(args.image_path,
                            img_h=img_size,
                            img_w=img_size)

        with torch.no_grad():
            o_t = model_t(sample.to(device)).cpu()

            data_np = torch.permute(sample, (0, 2, 3, 1)).numpy()
            tf_data = tf.convert_to_tensor(data_np)
            o_s  = model_s(input=tf_data)['output']
            o_s = torch.from_numpy(o_s)

            np.set_printoptions(linewidth=200)
            print('student : ', np.round(o_s[:, :args.n_vals], 4))
            print('teacher : ', np.round(o_t[:, :args.n_vals].numpy(), 4))
            print(f'cosine similarity: {cosine_similarity(o_s, o_t)}')

    