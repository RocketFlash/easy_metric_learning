import onnx
import torch
from pathlib import Path


def convert_onnx(
        traced_model,
        sample, 
        save_path, 
        model_name
    ):
    onnx_save_name = str(Path(save_path) / f'{model_name}.onnx')
    torch.onnx.export(  
        traced_model, 
        sample, 
        onnx_save_name, 
        input_names=['input'],
        output_names = ['output'], 
        dynamic_axes={
            'input' : {0 : 'batch_size'}, 
            'output' : {0 : 'batch_size'}
        },
        opset_version=16
    )

    model_onnx = onnx.load(onnx_save_name)
    onnx.checker.check_model(model_onnx, True)

    return model_onnx


def simplify_onnx(model_onnx):
    from onnxsim import simplify
    model_simp, check = simplify(model_onnx)
    assert check, "Simplified ONNX model could not be validated"
    return model_simp