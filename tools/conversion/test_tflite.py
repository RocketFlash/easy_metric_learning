import argparse
import os
from pathlib import Path

import sys

sys.path.insert(0, "../../")

import torch
from tqdm import tqdm

from src.utils import get_images_paths, get_sample


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke-test a TFLite embedding model")
    parser.add_argument("--model_path", required=True, help="path to .tflite model")
    parser.add_argument("--dataset_path", required=True, help="folder with images")
    parser.add_argument("--img_h", type=int, default=224)
    parser.add_argument("--img_w", type=int, default=224)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--cpu", action="store_true", help="hide CUDA devices from tensorflow"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=args.model_path)
    tf_lite_model = interpreter.get_signature_runner()

    image_paths = get_images_paths(Path(args.dataset_path))[: args.limit]
    for image_path in tqdm(image_paths):
        sample = get_sample(str(image_path), img_h=args.img_h, img_w=args.img_w)
        tf_sample = tf.convert_to_tensor(torch.permute(sample, (0, 2, 3, 1)).numpy())
        tf_lite_output = tf_lite_model(input=tf_sample)
        print(tf_lite_output["output"][:, :8])
