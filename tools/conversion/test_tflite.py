import tensorflow as tf
import torch
import sys
import os
from pathlib import Path
from tqdm import tqdm
sys.path.insert(0, '../../')
from src.utils import (get_images_paths,
                       get_sample)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    model_path = '/home/ubuntu/easy_metric_learning/work_dirs/full_arcface_openclip-ViT-B32_laion2b_m_0_6_s_30_backbone_lr_scaler_fold0/weights/arcface_openclip-ViT-B32_laion2b_im224_emb512.tf/arcface_openclip-ViT-B32_laion2b_im224_emb512_simp_float16.tflite'
    interpreter = tf.lite.Interpreter(model_path=model_path)
    tf_lite_model = interpreter.get_signature_runner()

    dataset_path = Path('/datasets/metric_learning/mksp_170x170')
    images_paths = get_images_paths(dataset_path)

    for image_path in tqdm(images_paths):
        sample = get_sample(str(image_path), img_h=224, img_w=224)
        tf_sample = tf.convert_to_tensor(torch.permute(sample, (0, 2, 3, 1)).numpy())
        tt_lite_output = tf_lite_model(input=tf_sample)
        print(tt_lite_output['output'][:, :8])