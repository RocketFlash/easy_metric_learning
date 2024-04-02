import argparse
import os

import omegaconf
import torch
import random
from torch.utils.mobile_optimizer import (optimize_for_mobile, 
                                          MobileOptimizerType)

import numpy as np
from pathlib import Path

from converter.onnx import (convert_onnx,
                            simplify_onnx)
torch.set_printoptions(profile="default")

from utils import (test_model, 
                   print_model_size,
                   get_sample,
                   get_images_paths,
                   generate_representative_dataset)

import sys
sys.path.insert(0, '../../')
from quantization import get_quantized_model
from src.model import load_emb_model_and_weights
from src.transform import get_transform
from src.utils import get_device


def parse_args():
    parser = argparse.ArgumentParser(
        description='Model conversion for mobile devices'
    )
    parser.add_argument(
        '--work_folder', 
        type=str, 
        default='', 
        help='path to trained model working directory'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='', 
        help='path to cfg.yaml'
    )
    parser.add_argument(
        '--weights', 
        type=str, 
        default='', 
        help='weights path'
    )
    parser.add_argument(
        '--save_path', 
        type=str, 
        default='', 
        help='save path'
    )
    parser.add_argument(
        '--device', 
        default='cpu', 
        type=str, 
        help='device for calculations'
    )
    parser.add_argument(
        '--dataset_path', 
        default='', 
        help='path to the calibration dataset'
    )
    parser.add_argument(
        '--q', 
        action='store_true', 
        help='Use quantization'
    )
    parser.add_argument(
        '--n_images_max', 
        default=1000,
        type=int, 
        help='Max number of images for quantization'
    )
    parser.add_argument(
        '--q_config', 
        default='x86',
        choices=[
            'x86',
            'qnnpack', 
        ],
        type=str, 
        help='q config'
    )
    parser.add_argument(
        '--q_dynamic', 
        action='store_true', 
        help='Use dynamic quantization'
    )
    parser.add_argument(
        '--profile', 
        action='store_true', 
        help='profile model'
    )
    parser.add_argument(
        '--mobile', 
        action='store_true', 
        help='convert model using mobile optimizers'
    )
    parser.add_argument(
        '--metal', 
        action='store_true', 
        help='convert model to metal backend'
    )
    parser.add_argument(
        '--nnapi', 
        action='store_true', 
        help='convert model to nnapi'
    )
    parser.add_argument(
        '--vulkan', 
        action='store_true', 
        help='convert model to vulkan backend'
    )
    parser.add_argument(
        '--onnx', 
        action='store_true', 
        help='convert model to onnx'
    )
    parser.add_argument(
        '--tf', 
        action='store_true', 
        help='convert model to tensorflow'
    )
    parser.add_argument(
        '--tf_q', 
        action='store_true', 
        help='do quantization for tensorflow model'
    )
    parser.add_argument(
        '--no_trace', 
        action='store_true', 
        help='do not trace model'
    )
    parser.add_argument(
        '--info', 
        default='', 
        help='additional info about the model'
    )

    args = parser.parse_args()

    if args.work_folder:
        args.work_folder = Path(args.work_folder)
        args.config = args.work_folder / 'config_train.yaml'

        if not args.config.is_file():
            args.config = args.work_folder / 'config.yaml'
        
        if not args.weights:
            args.weights = args.work_folder / 'last_emb.pt'
            
        if not args.save_path:
            args.save_path = args.work_folder / 'weights'

    return args


if __name__ == '__main__':
    args = parse_args()
    assert os.path.isfile(args.config)

    tab_str = '=================='
    n_times = 20
    print(f'{tab_str} Original model')
    if args.weights:
        print_model_size('Original model', args.weights)
    
    device = get_device(args.device)
    config = omegaconf.OmegaConf.load(args.config)

    save_path = Path(args.save_path)
    save_path.mkdir(exist_ok=True)

    model_name = config.run_name

    model = load_emb_model_and_weights(
        config_backbone=config.backbone,
        config_head=config.head,
        weights=args.weights,
        device=device,
    )
    model = model.eval().to(device)

    transform_test = get_transform(config.transform.test)

    image_paths = None
    if args.dataset_path:
        dataset_path = Path(args.dataset_path)
        image_paths = get_images_paths(dataset_path)
        random.Random(28).shuffle(image_paths)

        sample = get_sample(
            str(image_paths[0]), 
            transform=transform_test
        )
        sample_test = get_sample(
            str(image_paths[1]), 
            transform=transform_test
        )
    else:
        sample = torch.rand(1, 3, config.img_h, config.img_w)
        sample_test = torch.rand(1, 3, config.img_h, config.img_w)
    sample = sample.to(device)
    sample_test = sample_test.to(device)
    
    if not args.no_trace:
        test_model(
            model, 
            sample_test, 
            to_profile=args.profile,
            n_times=n_times
        )

        print(f'{tab_str} Torch model tracing')
        traced_model = torch.jit.trace(model, sample)
        traced_model_path = save_path / f'{model_name}_traced.pt'
        traced_model.save(traced_model_path)
        traced_model = traced_model.eval().to(device)
        print_model_size('Traced model', traced_model_path)
        test_model(
            traced_model, 
            sample_test, 
            to_profile=args.profile,
            n_times=n_times
        )
        
                    
    if args.q:
        print(f'{tab_str} Model quantization')
        assert image_paths is not None, "You need to set dataset_path for quantization"
        model_name += '_q'
        q_model_path = save_path / f'{model_name}.pt'
        model_q  = get_quantized_model(
            model,
            save_path,
            transform=transform_test,
            image_paths=image_paths,
            device=device,
            model_save_path=q_model_path,
            n_images_max=args.n_images_max,
            q_config=args.q_config,
            dynamic=args.q_dynamic
        )

        print_model_size('Qantized model', q_model_path)       
        test_model(
            model_q, 
            sample_test, 
            to_profile=args.profile,
            n_times=n_times
        )
        
        print(f'{tab_str} Quantized model tracing')
        traced_q_model = torch.jit.trace(model_q, sample)
        traced_q_model_path = save_path / f'{model_name}_traced.pt'
        traced_q_model.save(traced_q_model_path)
        traced_q_model = traced_q_model.eval().to(device)
        print_model_size('Qantized traced model', traced_q_model_path)
        test_model(
            traced_q_model, 
            sample_test, 
            to_profile=args.profile,
            n_times=n_times
        )
    else:
        model_q = model


    onnx_simp_save_name = str(save_path / f'{model_name}_simp.onnx')
    if args.onnx or args.tf:
        import onnx
        import onnxruntime as ort

        print(f'================== ONNX conversion')
        model_onnx_non_sim = convert_onnx(
            model, 
            sample, 
            save_path, 
            model_name
        )
        model_onnx = simplify_onnx(model_onnx_non_sim)
        onnx.save(model_onnx, onnx_simp_save_name)

        ort_session = ort.InferenceSession(
            onnx_simp_save_name,
            providers=['CPUExecutionProvider']
        )
        
        print_model_size('Onnx model', Path(onnx_simp_save_name))
        test_model(
            ort_session, 
            sample_test, 
            to_profile=args.profile,
            model_type='onnx',
            n_times=n_times
        )
        
        
    # if args.tf:
    #     assert os.path.isfile(onnx_simp_save_name), "ONNX file is not generated, generate it first"
    #     import tensorflow as tf
    #     print(f'================== TF conversion')
    #     import onnx2tf
    #     tf_save_name = str(save_path / f'{model_name}.tf')

    #     cind = None
    #     oiqt = False
    #     if args.tf_q:
    #         generate_representative_dataset(
    #             images_paths,
    #             image_size=(
    #                 img_size, 
    #                 img_size
    #             ),
    #             n_max=500,
    #             save_path=save_path
    #         )
    #         cind = [[
    #             'input', 
    #             str(save_path / 'calibdata.npy'), 
    #             np.array([[[[0.485, 0.456, 0.406]]]]).astype(np.float32), 
    #             np.array([[[[0.229, 0.224, 0.225]]]]).astype(np.float32)
    #         ]]
    #         oiqt = True
        
    #     try:
    #         onnx_file_name = Path(onnx_simp_save_name).stem
    #         tf_lite_save_name = str(Path(tf_save_name) / f'{onnx_file_name}_float32.tflite')
    #         onnx2tf.convert(
    #             input_onnx_file_path=onnx_simp_save_name,
    #             output_folder_path=tf_save_name,
    #             copy_onnx_input_output_names_to_tflite=True,
    #             non_verbose=True,
    #             custom_input_op_name_np_data_path=cind,
    #             output_integer_quantized_tflite=oiqt,
    #         )
    #     except:
    #         print('Can not convert simplified onnx to tf, convert non simplified version')
    #         tf_lite_save_name = str(Path(tf_save_name) / f'{model_name}_float32.tflite')
    #         onnx2tf.convert(
    #             input_onnx_file_path=str(Path(save_path) / f'{model_name}.onnx'),
    #             output_folder_path=tf_save_name,
    #             copy_onnx_input_output_names_to_tflite=True,
    #             non_verbose=True,
    #             custom_input_op_name_np_data_path=cind,
    #             output_integer_quantized_tflite=oiqt
    #         )

    #     interpreter = tf.lite.Interpreter(model_path=tf_lite_save_name)
    #     tf_lite_model = interpreter.get_signature_runner()

    #     tf_sample = tf.convert_to_tensor(torch.permute(sample, (0, 2, 3, 1)).numpy())
    #     tt_lite_output = tf_lite_model(input=tf_sample)

    #     o_t_o = tt_lite_output['output']
    #     print(o_t_o[:, :8])


    # if args.mobile:
    #     print(f'================== Mobile optimization')
    #     mob_blacklist = {
    #         MobileOptimizerType.CONV_BN_FUSION, 
    #         # MobileOptimizerType.FUSE_ADD_RELU,
    #         # MobileOptimizerType.HOIST_CONV_PACKED_PARAMS,
    #         # MobileOptimizerType.INSERT_FOLD_PREPACK_OPS,
    #         # MobileOptimizerType.REMOVE_DROPOUT
    #     }
    #     traced_model_optimized = optimize_for_mobile(traced_model,  mob_blacklist)
    #     traced_model_optimized._save_for_lite_interpreter(str(save_path / f'{model_name}_mobile.pt'))
    #     test_model(traced_model_optimized, sample, to_profile=args.profile, n_times=n_times)

    # if args.metal:
    #     # Optimize Metal
    #     print(f'===================== Metal optimization =============================')
    #     traced_model_optimized_metal = optimize_for_mobile(traced_model, mob_blacklist,  backend='metal')
    #     traced_model_optimized_metal._save_for_lite_interpreter(str(save_path / f'{model_name}_mobile_metal.pt'))

    # if args.nnapi:
    #     import torch.backends._nnapi.prepare
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


    
