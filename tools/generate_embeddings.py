import sys
sys.path.append("./")

import os
import argparse
import numpy as np
from pathlib import Path
import pandas as pd

from src.utils import load_ckp, load_config
from src.dataset import get_loader
from src.model import get_model_embeddings
from tqdm import tqdm
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_folder', type=str, default='', help='path to trained model working directory')
    parser.add_argument('--model_type', type=str, default='torch', help='path to cfg.yaml')
    parser.add_argument('--config', type=str, default='', help='path to cfg.yaml')
    parser.add_argument('--weights', type=str, default='', help='weights path')
    parser.add_argument('--save_path', type=str, default='', help='save path')
    parser.add_argument('--dataset_path', type=str, default='', help='path to dataset')
    parser.add_argument('--dataset_csv', type=str, default='', help='path to dataset csv file')
    parser.add_argument('--dataset_type', type=str, default='general', help='dataset type one of [general, cars, sop]')
    parser.add_argument('--bs',type=int, default=8, help='batch size')
    parser.add_argument('--device', type=str, default='cuda:0', help='select device')
    parser.add_argument('--n_jobs', type=int, default=4, help='number of parallel jobs')
    parser.add_argument('--img_size',type=int, default=170, help='input image size')
    parser.add_argument('--emb_size',type=int, default=512, help='embeddings size')
    parser.add_argument('--use_bboxes', action='store_true', help='use regions from bboxes')
    parser.add_argument('--data_type', type=str, default='general', help='preprocssing data type one of [general, clip]')
    
    return parser.parse_args()


def main(CONFIGS, args):
    weights = args.weights
    dataset_csv = args.dataset_csv
    dataset_path = args.dataset_path
    save_path = args.save_path 
    bs = args.bs
    emb_size = args.emb_size
    n_workers=args.n_jobs

    if dataset_path:
        CONFIGS["DATA"]["DIR"] = dataset_path

    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True)

    device = torch.device(CONFIGS['GENERAL']['DEVICE'])

    df_dtype = {
        'label': str,
        'file_name': str,
        'width': int,
        'height': int,
        'is_test':int
    }  

    df = pd.read_csv(dataset_csv, 
                     dtype=df_dtype)
    
    eval_status = None
    if args.dataset_type in ['cars', 'sop', 'cub']:
        df = df[df['is_test'] == 1].reset_index()
    elif args.dataset_type in ['inshop']:
        df = df[df['is_test'] == 1].reset_index()
        eval_status = df['evaluation_status'].values
    
    CONFIGS['DATA']['USE_CATEGORIES'] = False
    data_loader, dataset = get_loader(
        df,
        data_config=CONFIGS["DATA"],
        split='val',
        test=True,
        batch_size=bs,
        num_thread=n_workers,
        label_column='label',
        fname_column='file_name',
        return_filenames=True,
        transform_name='no_aug',
        use_bboxes=args.use_bboxes
    )
    ids_to_labels = dataset.get_ids_to_labels()

    if args.model_type=='torch':
        model = get_model_embeddings(model_config=CONFIGS['MODEL'])
        model = load_ckp(weights, model, emb_model_only=True)
        model.to(device)
        model.eval()
    elif args.model_type=='traced':
        model = torch.jit.load(weights)
        model.to(device)
        model.eval()
    elif args.model_type=='onnx':
        import onnxruntime as ort
        model = ort.InferenceSession(
            weights,
            providers=[
                'CUDAExecutionProvider', 
                'CPUExecutionProvider'
            ]
        )
    elif args.model_type in ['tf_32', 'tf_16', 'tf_int', 'tf_dyn']:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=weights)
        model = interpreter.get_signature_runner()
    elif args.model_type=='tf_full_int':
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=weights)
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        model = interpreter.get_signature_runner()
    
    embeddings = np.zeros((len(df), emb_size), dtype=np.float32)
    labels = np.zeros(len(df), dtype=object)
    file_names = np.zeros(len(df), dtype=object)

    tqdm_bar = tqdm(data_loader, total=int(len(data_loader)))

    with torch.no_grad():
        index = 0
        for batch_index, (data, targets, file_nms) in enumerate(tqdm_bar):
            if args.model_type in ['torch', 'traced']:
                data = data.to(device)
                output = model(data)
            elif args.model_type=='onnx':
                output = model.run( None, {"input": data.numpy()})[0]
            elif args.model_type in ['tf_32', 'tf_16', 'tf_int', 'tf_dyn']:
                data_np = torch.permute(data, (0, 2, 3, 1)).numpy()
                tf_data = tf.convert_to_tensor(data_np)
                output  = model(input=tf_data)['output']
            elif args.model_type=='tf_full_int':
                data_np = torch.permute(data, (0, 2, 3, 1)).numpy()
                input_scale, input_zero_point = input_details["quantization"]
                output_scale, output_zero_point = output_details['quantization']
                tf_sample_int8 = data_np / input_scale + input_zero_point
                tf_sample_int8 = tf_sample_int8.astype(input_details["dtype"])
                tf_sample_int8 = tf.convert_to_tensor(tf_sample_int8)
                output = model(input=tf_sample_int8)
                output = output_scale * (output['output'].astype(np.float32) - output_zero_point)
                
            batch_size = output.shape[0]

            if torch.is_tensor(output):
                output = output.cpu().numpy()

            lbls = [ids_to_labels[t] for t in targets.cpu().numpy()]
            embeddings[index:(index+batch_size), :] = output
            labels[index:(index+batch_size)] = lbls
            file_names[index:(index+batch_size)] = file_nms
            index += batch_size
    
    non_empty_rows_mask = file_names != ''
    embeddings = embeddings[non_empty_rows_mask]
    labels = labels[non_empty_rows_mask]
    file_names = file_names[non_empty_rows_mask]

    np.savez(save_path / 'embeddings.npz', 
             embeddings=embeddings, 
             labels=labels,
             file_names=file_names,
             eval_status=eval_status)

if __name__ == '__main__':
    args = parse_args()

    if args.work_folder:
        args.work_folder = Path(args.work_folder)
        args.config = args.work_folder / 'config.yml'

        assert os.path.isfile(args.config)
        CONFIGS = load_config(args.config)

        if not args.weights:
            if args.model_type:
                margin_type  = CONFIGS['MODEL']['MARGIN_TYPE']
                encoder_type = CONFIGS['MODEL']['ENCODER_NAME']
                img_size     = CONFIGS['DATA']['IMG_SIZE']
                emb_size     = CONFIGS['MODEL']['EMBEDDINGS_SIZE']
                model_name = f'{margin_type}_{encoder_type}_im{img_size}_emb{emb_size}'
                WEIGHTS_PATH = args.work_folder / 'weights'
                WEIGHTS_PATH_TF = WEIGHTS_PATH  / f'{model_name}.tf'

                if args.model_type=='torch':
                    WEIGHTS = args.work_folder / 'best_emb.pt'
                elif args.model_type=='traced':
                    WEIGHTS = WEIGHTS_PATH / f'{model_name}_traced.pt'
                elif args.model_type=='onnx':
                    WEIGHTS = WEIGHTS_PATH / f'{model_name}_simp.onnx'
                elif args.model_type=='tf_32':
                    WEIGHTS = str(WEIGHTS_PATH_TF / f'{model_name}_simp_float32.tflite')
                elif args.model_type=='tf_16':
                    WEIGHTS = str(WEIGHTS_PATH_TF / f'{model_name}_simp_float16.tflite')
                elif args.model_type=='tf_dyn':
                    WEIGHTS = str(WEIGHTS_PATH_TF  / f'{model_name}_simp_dynamic_range_quant.tflite')
                elif args.model_type=='tf_int':
                    WEIGHTS = str(WEIGHTS_PATH_TF / f'{model_name}_simp_integer_quant.tflite')
                elif args.model_type=='tf_full_int':
                    WEIGHTS = str(WEIGHTS_PATH_TF / f'{model_name}_simp_full_integer_quant.tflite')
                else:
                    raise ValueError('model_type malue must be one of [torch, traced, onnx, tf_32, tf_16, tf_dyn, tf_int, tf_full_int]')
                args.weights = WEIGHTS
            else:
                args.weights = args.work_folder / 'best_emb.pt'

        dataset_name = Path(args.dataset_path).name
        if not args.save_path:
            args.save_path = args.work_folder / 'embeddings' / dataset_name
    else:
        if os.path.isfile(args.config):
            CONFIGS = load_config(args.config)
        else:
            CONFIGS = {
                "GENERAL" : {
                    "DEVICE" : args.device,
                    "WORKERS" : args.n_jobs
                },
                "DATA" : {
                    'DATASET_TYPE' : 'simple',                                     
                    'DATA_TYPE' : args.data_type,
                    'IMG_SIZE'  : args.img_size
                }
            }

    save_path = Path(args.save_path)
    save_path.mkdir(exist_ok=True, parents=True)

    main(CONFIGS, args)