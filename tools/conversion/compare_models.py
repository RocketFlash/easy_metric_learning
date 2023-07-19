import argparse

from pathlib import Path
import os
import cv2
import copy
from pprint import pprint

from tqdm import tqdm
import tensorflow as tf
import torch
import numpy as np
import onnxruntime as ort
torch.set_printoptions(profile="full")

import random
import sys
sys.path.insert(0, '../../')
from src.model import get_model_embeddings
from src.utils import (get_device,
                       load_ckp, 
                       get_images_paths,
                       load_config,
                       get_sample)
from torch.nn.functional import cosine_similarity

def parse_args():
    parser = argparse.ArgumentParser(description='Model conversion for mobile devices')
    parser.add_argument('--work_folder', type=str, default='', help='path to trained model working directory')
    parser.add_argument('--dataset_path', default='', help='path to the dataset')
    parser.add_argument('--image_path', type=str, default='', help='image path')
    parser.add_argument('--n_vals', type=int, default=8, help='number of elements to show')
    parser.add_argument('--n_samples', type=int, default=200, help='number of samples to compare')
    parser.add_argument('--device', default='cpu', type=str, help='device for calculations')
    return parser.parse_args()


def get_size(file_path, unit='bytes'):
    file_size = os.path.getsize(file_path)
    exponents_map = {'bytes': 0, 'kb': 1, 'mb': 2, 'gb': 3}
    if unit not in exponents_map:
        raise ValueError("Must select from \
        ['bytes', 'kb', 'mb', 'gb']")
    else:
        size = file_size / 1024 ** exponents_map[unit]
        return round(size, 3)

def get_mean(arr):
    return sum(arr)/len(arr)

if __name__ == '__main__':
    args = parse_args()

    assert args.work_folder
    args.work_folder = Path(args.work_folder)
    config = args.work_folder / 'config.yml'
    weights = args.work_folder / 'best_emb.pt'
    WEIGHTS_PATH = args.work_folder / 'weights'
    
    device = get_device(args.device)
    assert os.path.isfile(config)
    CONFIGS  = load_config(config)
    margin_type  = CONFIGS['MODEL']['MARGIN_TYPE']
    encoder_type = CONFIGS['MODEL']['ENCODER_NAME']
    img_size     = CONFIGS['DATA']['IMG_SIZE']
    emb_size     = CONFIGS['MODEL']['EMBEDDINGS_SIZE']

    model_name = f'{margin_type}_{encoder_type}_im{img_size}_emb{emb_size}'

    WEIGHTS = args.work_folder / 'best_emb.pt'
    WEIGHTS_TRACED = WEIGHTS_PATH / f'{model_name}_traced.pt'
    WEIGHTS_ONNX = WEIGHTS_PATH / f'{model_name}_simp.onnx'
    WEIGHTS_PATH_TF = WEIGHTS_PATH  / f'{model_name}.tf'
    WEIGHTS_TF_FLOAT_32 = str(WEIGHTS_PATH_TF / f'{model_name}_simp_float32.tflite')
    WEIGHTS_TF_FLOAT_16 = str(WEIGHTS_PATH_TF / f'{model_name}_simp_float16.tflite')
    WEIGHTS_TF_DYNAMIC = str(WEIGHTS_PATH_TF  / f'{model_name}_simp_dynamic_range_quant.tflite')
    WEIGHTS_TF_INT = str(WEIGHTS_PATH_TF / f'{model_name}_simp_integer_quant.tflite')
    WEIGHTS_TF_FULL_INT = str(WEIGHTS_PATH_TF / f'{model_name}_simp_full_integer_quant.tflite')

    model_size_orig   = get_size(WEIGHTS, 'mb')
    model_size_traced = get_size(WEIGHTS_TRACED, 'mb')
    model_size_onnx   = get_size(WEIGHTS_ONNX, 'mb')
    model_size_tf32   = get_size(WEIGHTS_TF_FLOAT_32, 'mb')
    model_size_tf16   = get_size(WEIGHTS_TF_FLOAT_16, 'mb')
    model_size_dyn    = get_size(WEIGHTS_TF_DYNAMIC, 'mb')
    model_size_int    = get_size(WEIGHTS_TF_INT, 'mb')
    model_size_fint   = get_size(WEIGHTS_TF_FULL_INT, 'mb')


    print('========= load model torch')
    model = get_model_embeddings(model_config=CONFIGS['MODEL'])
    model = load_ckp(weights, model, emb_model_only=True)
    model = model.eval().to(device)
   
    print('========= load model traced')
    model_traced = torch.jit.load(WEIGHTS_TRACED)
    model_traced = model_traced.eval().to(device)

    print('========= load onnx model')
    ort_session = ort.InferenceSession(WEIGHTS_ONNX,
                                       providers=['CUDAExecutionProvider', 
                                                  'CPUExecutionProvider'])

    print('========= load tf model (float32)')
    interpreter = tf.lite.Interpreter(model_path=WEIGHTS_TF_FLOAT_32)
    tf_lite_model_float32 = interpreter.get_signature_runner()

    print('========= load tf model (float16)')
    interpreter = tf.lite.Interpreter(model_path=WEIGHTS_TF_FLOAT_16)
    tf_lite_model_float16 = interpreter.get_signature_runner()

    print('========= load tf model (dynamic range)')
    interpreter = tf.lite.Interpreter(model_path=WEIGHTS_TF_DYNAMIC)
    tf_lite_model_dynamic = interpreter.get_signature_runner()

    print('========= load tf model (int8)')
    interpreter = tf.lite.Interpreter(model_path=WEIGHTS_TF_INT)
    tf_lite_model_int = interpreter.get_signature_runner()

    print('========= load tf model (full int8)')
    interpreter = tf.lite.Interpreter(model_path=WEIGHTS_TF_FULL_INT)
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    tf_lite_model_full_int = interpreter.get_signature_runner()
    

    if args.dataset_path:
        dataset_path = Path(args.dataset_path)
        images_paths = get_images_paths(dataset_path)
        random.shuffle(images_paths)
        images_paths = images_paths[:args.n_samples]

        cos_sim_traced = []
        cos_sim_onnx   = []
        cos_sim_tf32   = []
        cos_sim_tf16   = []
        cos_sim_dyn    = []
        cos_sim_int    = []
        cos_sim_fint   = []

        for image_path in tqdm(images_paths):
            sample = get_sample(str(image_path),
                                img_h=img_size,
                                img_w=img_size)
            
            with torch.no_grad():
                sample_np = torch.permute(sample, (0, 2, 3, 1)).numpy()
                tf_sample = tf.convert_to_tensor(sample_np)
                
                input_scale, input_zero_point = input_details["quantization"]
                output_scale, output_zero_point = output_details['quantization']
                tf_sample_int8 = sample_np / input_scale + input_zero_point
                tf_sample_int8 = tf_sample_int8.astype(input_details["dtype"])
                tf_sample_int8 = tf.convert_to_tensor(tf_sample_int8)

                sample = sample.to(device)
                o_orig = model(sample).cpu()
                o_traced = model_traced(sample).cpu()
                o_onnx = ort_session.run( None, {"input": sample.cpu().numpy()})
                o_tf_lite_f32  = tf_lite_model_float32(input=tf_sample)
                o_tf_lite_f16  = tf_lite_model_float16(input=tf_sample)
                o_tf_lite_dyn  = tf_lite_model_dynamic(input=tf_sample)
                o_tf_lite_int  = tf_lite_model_int(input=tf_sample)
                o_tf_lite_fint = tf_lite_model_full_int(input=tf_sample_int8)
                o_tf_lite_fint = output_scale * (o_tf_lite_fint['output'].astype(np.float32) - output_zero_point)
                
                cos_sim_traced_i = cosine_similarity(o_orig, o_traced)
                cos_sim_onnx_i   = cosine_similarity(o_orig, torch.from_numpy(o_onnx[0]))
                cos_sim_tf32_i   = cosine_similarity(o_orig, torch.from_numpy(o_tf_lite_f32['output']))
                cos_sim_tf16_i   = cosine_similarity(o_orig, torch.from_numpy(o_tf_lite_f16['output']))
                cos_sim_dyn_i    = cosine_similarity(o_orig, torch.from_numpy(o_tf_lite_dyn['output']))
                cos_sim_int_i    = cosine_similarity(o_orig, torch.from_numpy(o_tf_lite_int['output']))
                cos_sim_fint_i   = cosine_similarity(o_orig, torch.from_numpy(o_tf_lite_fint))

                cos_sim_traced.append(cos_sim_traced_i)
                cos_sim_onnx.append(cos_sim_onnx_i)
                cos_sim_tf32.append(cos_sim_tf32_i)
                cos_sim_tf16.append(cos_sim_tf16_i)
                cos_sim_dyn.append(cos_sim_dyn_i)
                cos_sim_int.append(cos_sim_int_i)
                cos_sim_fint.append(cos_sim_fint_i)

        np.set_printoptions(linewidth=200)
        print('========= Model outputs')
        print('original : ', np.round(o_orig[:, :args.n_vals].numpy(), 4))
        print('traced   : ', np.round(o_traced[:, :args.n_vals].numpy(), 4))
        print('onnx     : ', np.round(o_onnx[0][:, :args.n_vals], 4))
        print('tf32     : ', np.round(o_tf_lite_f32['output'][:, :args.n_vals], 4))
        print('tf16     : ', np.round(o_tf_lite_f16['output'][:, :args.n_vals], 4))
        print('tf_dyn   : ', np.round(o_tf_lite_dyn['output'][:, :args.n_vals], 4))
        print('tf_int   : ', np.round(o_tf_lite_int['output'][:, :args.n_vals], 4))
        print('tf_fint  : ', np.round(o_tf_lite_fint[:, :args.n_vals], 4))
        print(f'avg cosine similarity traced : {get_mean(cos_sim_traced)}')
        print(f'avg cosine similarity onnx   : {get_mean(cos_sim_onnx)}')
        print(f'avg cosine similarity tf_f32 : {get_mean(cos_sim_tf32)}')
        print(f'avg cosine similarity tf_f16 : {get_mean(cos_sim_tf16)}')
        print(f'avg cosine similarity tf_dyn : {get_mean(cos_sim_dyn)}')
        print(f'avg cosine similarity tf_int : {get_mean(cos_sim_int)}')
        print(f'avg cosine similarity tf_fint: {get_mean(cos_sim_fint)}')
    
    if args.image_path:
        sample = get_sample(args.image_path,
                            img_h=img_size,
                            img_w=img_size)
        with torch.no_grad():
            sample_np = torch.permute(sample, (0, 2, 3, 1)).numpy()
            tf_sample = tf.convert_to_tensor(sample_np)
            
            input_scale, input_zero_point = input_details["quantization"]
            output_scale, output_zero_point = output_details['quantization']
            tf_sample_int8 = sample_np / input_scale + input_zero_point
            tf_sample_int8 = tf_sample_int8.astype(input_details["dtype"])
            tf_sample_int8 = tf.convert_to_tensor(tf_sample_int8)
        
            o_orig = model(sample)
            o_traced = model_traced(sample)
            o_onnx = ort_session.run( None, {"input": sample.numpy()})
            o_tf_lite_f32  = tf_lite_model_float32(input=tf_sample)
            o_tf_lite_f16  = tf_lite_model_float16(input=tf_sample)
            o_tf_lite_dyn  = tf_lite_model_dynamic(input=tf_sample)
            o_tf_lite_int  = tf_lite_model_int(input=tf_sample)
            o_tf_lite_fint = tf_lite_model_full_int(input=tf_sample_int8)
            o_tf_lite_fint = output_scale * (o_tf_lite_fint['output'].astype(np.float32) - output_zero_point)

            np.set_printoptions(linewidth=150)
            print('========= Model outputs')
            print('original : ', np.round(o_orig[:, :args.n_vals].numpy(), 4))
            print('traced   : ', np.round(o_traced[:, :args.n_vals].numpy(), 4))
            print('onnx     : ', np.round(o_onnx[0][:, :args.n_vals], 4))
            print('tf_f32   : ', np.round(o_tf_lite_f32['output'][:, :args.n_vals], 4))
            print('tf_f16   : ', np.round(o_tf_lite_f16['output'][:, :args.n_vals], 4))
            print('tf_dyn   : ', np.round(o_tf_lite_dyn['output'][:, :args.n_vals], 4))
            print('tf_int   : ', np.round(o_tf_lite_int['output'][:, :args.n_vals], 4))
            print('tf_fint  : ', np.round(o_tf_lite_fint[:, :args.n_vals], 4))

    print('========= Models sizes')
    print('original : ', model_size_orig, 'mb')
    print('traced   : ', model_size_traced, 'mb')
    print('onnx     : ', model_size_onnx, 'mb')
    print('tf_f32   : ', model_size_tf32, 'mb')
    print('tf_f16   : ', model_size_tf16, 'mb')
    print('tf_dyn   : ', model_size_dyn, 'mb')
    print('tf_int   : ', model_size_int, 'mb')
    print('tf_fint  : ', model_size_fint, 'mb')

            
        

        
        

        