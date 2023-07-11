import argparse
import os

import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
from torch.utils.mobile_optimizer import optimize_for_mobile, MobileOptimizerType
import torch.backends._nnapi.prepare
import copy
import numpy as np
from pathlib import Path
import onnx
import onnxruntime as ort
torch.set_printoptions(profile="default")

from utils import (test_model, 
                   generate_representative_dataset)
from quantization import get_quantization

import sys
sys.path.insert(0, '../../')
# from quantization import get_quantization
from src.model import get_model_embeddings
from src.utils import (get_device,
                       load_ckp, 
                       load_config,
                       get_sample,
                       get_images_paths)


def parse_args():
    parser = argparse.ArgumentParser(description='Model conversion for mobile devices')
    parser.add_argument('--work_folder', type=str, default='', help='path to trained model working directory')
    parser.add_argument('--config', type=str, default='', help='path to cfg.yaml')
    parser.add_argument('--weights', type=str, default='', help='weights path')
    parser.add_argument('--save_path', type=str, default='', help='save path')
    parser.add_argument('--device', default='cpu', type=str, help='device for calculations')
    parser.add_argument('--dataset_path', default='', help='path to the calibration dataset')
    parser.add_argument('--q', action='store_true', help='Use quantization')
    parser.add_argument('--backbone', default='mobilenetv2', type=str, help='backbone type')
    parser.add_argument('--profile', action='store_true', help='profile model')
    parser.add_argument('--mobile', action='store_true', help='convert model using mobile optimizers')
    parser.add_argument('--metal', action='store_true', help='convert model to metal backend')
    parser.add_argument('--nnapi', action='store_true', help='convert model to nnapi')
    parser.add_argument('--vulkan', action='store_true', help='convert model to vulkan backend')
    parser.add_argument('--onnx', action='store_true', help='convert model to onnx')
    parser.add_argument('--tf', action='store_true', help='convert model to tensorflow')
    parser.add_argument('--tf_q', action='store_true', help='do quantization for tensorflow model')
    parser.add_argument('--no_trace', action='store_true', help='do not trace model')
    parser.add_argument('--info', default='', help='additional info about the model')
    return parser.parse_args()


def convert_onnx(traced_model,
                 sample, 
                 save_path, 
                 model_name):
    onnx_save_name = str(Path(save_path) / f'{model_name}.onnx')
    torch.onnx.export(  traced_model, 
                        sample, 
                        onnx_save_name, 
                        input_names=['input'],
                        output_names = ['output'], 
                        dynamic_axes={
                                      'input' : {0 : 'batch_size'}, 
                                      'output' : {0 : 'batch_size'}
                                      },
                        opset_version=16)

    model_onnx = onnx.load(onnx_save_name)
    onnx.checker.check_model(model_onnx, True)

    return model_onnx


def simplify_onnx(model_onnx):
    from onnxsim import simplify
    model_simp, check = simplify(model_onnx)
    assert check, "Simplified ONNX model could not be validated"
    return model_simp


if __name__ == '__main__':
    args = parse_args()

    if args.work_folder:
        args.work_folder = Path(args.work_folder)
        args.config = args.work_folder / 'config.yml'
        args.weights = args.work_folder / 'best_emb.pt'
        if not args.save_path:
            args.save_path = args.work_folder / 'weights'
    
    device = get_device(args.device)
    assert os.path.isfile(args.config)
    CONFIGS = load_config(args.config)

    save_path = Path(args.save_path)
    save_path.mkdir(exist_ok=True)

    margin_type  = CONFIGS['MODEL']['MARGIN_TYPE']
    encoder_type = CONFIGS['MODEL']['ENCODER_NAME']
    img_size     = CONFIGS['DATA']['IMG_SIZE']
    emb_size     = CONFIGS['MODEL']['EMBEDDINGS_SIZE']

    model_name = f'{margin_type}_{encoder_type}_im{img_size}_emb{emb_size}'

    model = get_model_embeddings(model_config=CONFIGS['MODEL'])
    model = load_ckp(args.weights, model, emb_model_only=True)
    model = model.eval().to(device)

    if args.dataset_path:
        dataset_path = Path(args.dataset_path)
        images_paths = get_images_paths(dataset_path)
        
        sample = get_sample(str(images_paths[0]), 
                                img_size, 
                                img_size)
    else:
        sample = torch.rand(1, 3, img_size, img_size)

    sample = sample.to(device)

    mob_blacklist = {
                        MobileOptimizerType.CONV_BN_FUSION, 
                        # MobileOptimizerType.FUSE_ADD_RELU,
                        # MobileOptimizerType.HOIST_CONV_PACKED_PARAMS,
                        # MobileOptimizerType.INSERT_FOLD_PREPACK_OPS,
                        # MobileOptimizerType.REMOVE_DROPOUT
                    }
    
    if not args.no_trace:
        print(f'================== Full model')
        test_model(model, 
                   sample, 
                   to_profile=args.profile)
                    
        if args.q:
            print(f'================== Model quantization')
            model_name += '_q'
            model_q  = get_quantization(model,
                                        save_path,
                                        image_size=(img_size, 
                                                    img_size),
                                        images_path=images_paths,
                                        device=device,
                                        model_name=model_name)
                    
            test_model(model_q, 
                       sample, 
                       to_profile=args.profile)
        else:
            model_q = model

        print(f'================== Model tracing')
        traced_model = torch.jit.trace(model_q, sample)
        traced_model.save(save_path / f'{model_name}_traced.pt')
        traced_model = traced_model.eval().to(device)
        test_model(traced_model, sample, to_profile=args.profile)

    onnx_simp_save_name = str(save_path / f'{model_name}_simp.onnx')
    if args.onnx or args.tf:
        print(f'================== ONNX conversion')
        model_onnx_non_sim = convert_onnx(traced_model, 
                                          sample, 
                                          save_path, 
                                          model_name)
        model_onnx = simplify_onnx(model_onnx_non_sim)
        onnx.save(model_onnx, onnx_simp_save_name)

        ort_session = ort.InferenceSession(onnx_simp_save_name,
                                           providers=['CPUExecutionProvider'])
        outputs = ort_session.run( None, {"input": sample.cpu().numpy()})
        print(outputs[0][:, :8])

    if args.tf:
        assert os.path.isfile(onnx_simp_save_name), "ONNX file is not generated, generate it first"
        import tensorflow as tf
        print(f'================== TF conversion')
        import onnx2tf
        tf_save_name = str(save_path / f'{model_name}.tf')

        cind = None
        oiqt = False
        if args.tf_q:
            generate_representative_dataset(images_paths,
                                            image_size=(img_size, 
                                                        img_size),
                                            n_max=500,
                                            save_path=save_path)
            cind = [
                        [
                        'input', 
                        str(save_path / 'calibdata.npy'), 
                        np.array([[[[0.485, 0.456, 0.406]]]]).astype(np.float32), 
                        np.array([[[[0.229, 0.224, 0.225]]]]).astype(np.float32)
                        ]
                    ]
            oiqt = True
        
        try:
            onnx_file_name = Path(onnx_simp_save_name).stem
            tf_lite_save_name = str(Path(tf_save_name) / f'{onnx_file_name}_float32.tflite')
            onnx2tf.convert(
                            input_onnx_file_path=onnx_simp_save_name,
                            output_folder_path=tf_save_name,
                            copy_onnx_input_output_names_to_tflite=True,
                            non_verbose=True,
                            custom_input_op_name_np_data_path=cind,
                            output_integer_quantized_tflite=oiqt,

                        )
        except:
            print('Can not convert simplified onnx to tf, convert non simplified version')
            tf_lite_save_name = str(Path(tf_save_name) / f'{model_name}_float32.tflite')
            onnx2tf.convert(
                            input_onnx_file_path=str(Path(save_path) / f'{model_name}.onnx'),
                            output_folder_path=tf_save_name,
                            copy_onnx_input_output_names_to_tflite=True,
                            non_verbose=True,
                            custom_input_op_name_np_data_path=cind,
                            output_integer_quantized_tflite=oiqt
                        )

        interpreter = tf.lite.Interpreter(model_path=tf_lite_save_name)
        tf_lite_model = interpreter.get_signature_runner()

        tf_sample = tf.convert_to_tensor(torch.permute(sample, (0, 2, 3, 1)).numpy())
        tt_lite_output = tf_lite_model(input=tf_sample)

        o_t_o = tt_lite_output['output']
        print(o_t_o[:, :8])


    if args.mobile:
        print(f'================== Mobile optimization')
        traced_model_optimized = optimize_for_mobile(traced_model,  mob_blacklist)
        traced_model_optimized._save_for_lite_interpreter(str(save_path / f'{model_name}_mobile.pt'))
        test_model(traced_model_optimized, sample, to_profile=args.profile)

    # if args.metal:
    #     # Optimize Metal
    #     print(f'===================== Metal optimization =============================')
    #     traced_model_optimized_metal = optimize_for_mobile(traced_model, mob_blacklist,  backend='metal')
    #     traced_model_optimized_metal._save_for_lite_interpreter(str(save_path / f'{model_name}_mobile_metal.pt'))

    # if args.nnapi:
    #     print(f'===================== NNAPI optimization =============================')
    #     input_tensor = sample.contiguous(memory_format=torch.channels_last)
    #     input_tensor.nnapi_nhwc = True
    #     with torch.no_grad():
    #         traced_nnapi_model = torch.jit.trace(model_q, input_tensor)
    #     nnapi_model = torch.backends._nnapi.prepare.convert_model_to_nnapi(traced_nnapi_model, input_tensor)
    #     nnapi_model._save_for_lite_interpreter(save_path / f'{model_name}_nnapi.pt')


    # if args.vulkan:
    #     print(f'===================== Vulkan optimization =============================')
    #     model = get_model(num_angle=args.num_angle,
    #                         num_rho=args.num_rho,
    #                         backbone=args.backbone,
    #                         device=device,
    #                         hough_cuda=False,
    #                         pretrained=False, 
    #                         quantization=True,
    #                         is_vulkan=True).eval().to(device)
        
    #     load_weights(model, 
    #                 args.weights, 
    #                 device)
        
    #     model_full = DHT_full(copy.deepcopy(model), 
    #                         num_rho=args.num_rho, 
    #                         num_angle=args.num_angle, 
    #                         threshold=args.threshold,
    #                         n_cc=args.n_cc,
    #                         conversion=True,
    #                         stp=1).eval().to(device)
        
    #     profile_model(model_full, sample)
    #     print(model_full(sample))
        
    #     traced_model = torch.jit.trace(model_full, sample)
    #     traced_model.save(Path(args.save_path) / f'{model_name}_traced.pt')
    #     traced_model.eval()

    #     traced_model_optimized_vulkan = optimize_for_mobile(traced_model, {MobileOptimizerType.CONV_BN_FUSION, },  backend='vulkan')
    #     traced_model_optimized_vulkan._save_for_lite_interpreter(str(Path(args.save_path) / f'{model_name}_mobile_vulkan_q.pt'))

    #     with torch.no_grad():
    #         o_t_o = traced_model_optimized_vulkan(sample.to(device='vulkan'))
    #     print('Quantized mobile model output: ')
    #     profile_model(traced_model_optimized_vulkan, sample.to(device='vulkan'))
    #     print(o_t_o[0])
    #     print(o_t_o[1])
    #     print(o_t_o[2])


    
