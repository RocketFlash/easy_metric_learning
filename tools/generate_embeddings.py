import sys

sys.path.append("./")

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from src.data import get_loader
from src.model import load_emb_model_and_weights
from src.transform import get_transform
from src.utils import get_device, load_model_except_torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--work_folder",
        type=str,
        default="",
        help="path to trained model working directory",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="torch",
        help="one of [torch, traced, onnx, tf_32, tf_16, tf_dyn, tf_int, tf_full_int]",
    )
    parser.add_argument(
        "--config", type=str, default="", help="path to config_train.yaml"
    )
    parser.add_argument("--weights", type=str, default="", help="weights path")
    parser.add_argument("--save_path", type=str, default="", help="save path")
    parser.add_argument(
        "--dataset_path", type=str, default="", help="path to dataset root"
    )
    parser.add_argument(
        "--dataset_csv", type=str, default="", help="path to dataset csv file"
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="general",
        help="one of [general, cars, sop, cub, inshop]",
    )
    parser.add_argument("--bs", type=int, default=8, help="batch size")
    parser.add_argument("--device", type=str, default="", help="select device")
    parser.add_argument(
        "--n_jobs", type=int, default=4, help="number of dataloader workers"
    )
    parser.add_argument(
        "--img_size", type=int, default=224, help="fallback input image size"
    )
    parser.add_argument(
        "--emb_size", type=int, default=512, help="fallback embeddings size"
    )
    parser.add_argument(
        "--use_bboxes", action="store_true", help="use regions from bboxes"
    )
    return parser.parse_args()


def find_config(work_folder):
    for name in ("config_train.yaml", "config_test.yaml", "config.yaml", "config.yml"):
        path = work_folder / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No config file found in {work_folder}")


def resolve_args(args):
    if args.work_folder:
        work_folder = Path(args.work_folder)
        args.config = Path(args.config) if args.config else find_config(work_folder)
        if not args.weights:
            if args.model_type == "torch":
                for name in ("best_emb.pt", "last_emb.pt"):
                    candidate = work_folder / name
                    if candidate.is_file():
                        args.weights = candidate
                        break
            else:
                config = OmegaConf.load(args.config)
                weights_dir = work_folder / "weights"
                model_name = config.run_name
                weight_names = {
                    "traced": f"{model_name}_traced.pt",
                    "onnx": f"{model_name}_simp.onnx",
                    "tf_32": f"{model_name}.tf/{model_name}_simp_float32.tflite",
                    "tf_16": f"{model_name}.tf/{model_name}_simp_float16.tflite",
                    "tf_dyn": f"{model_name}.tf/{model_name}_simp_dynamic_range_quant.tflite",
                    "tf_int": f"{model_name}.tf/{model_name}_simp_integer_quant.tflite",
                    "tf_full_int": f"{model_name}.tf/{model_name}_simp_full_integer_quant.tflite",
                }
                if args.model_type in weight_names:
                    args.weights = weights_dir / weight_names[args.model_type]

        if not args.save_path and args.dataset_path:
            args.save_path = work_folder / "embeddings" / Path(args.dataset_path).name

    if not args.config:
        raise ValueError("--config or --work_folder is required")
    if not args.weights:
        raise ValueError("--weights could not be resolved")
    if not args.dataset_path or not args.dataset_csv:
        raise ValueError("--dataset_path and --dataset_csv are required")
    if not args.save_path:
        raise ValueError("--save_path is required when --work_folder is not set")

    args.config = Path(args.config)
    args.weights = Path(args.weights)
    args.save_path = Path(args.save_path)
    return args


def read_annotations(args):
    df = pd.read_csv(
        args.dataset_csv,
        dtype={
            "label": str,
            "file_name": str,
            "width": "Int64",
            "height": "Int64",
            "is_test": "Int64",
        },
    )
    eval_status = None
    if args.dataset_type in ["cars", "sop", "cub", "inshop"] and "is_test" in df:
        df = df[df["is_test"] == 1].reset_index(drop=True)
        if args.dataset_type == "inshop" and "evaluation_status" in df:
            eval_status = df["evaluation_status"].values
    return df, eval_status


def build_dataloader(config, args, df):
    transform_test = get_transform(config.transform.test)
    dataset_config = SimpleNamespace(
        type="simple",
        label_column="label",
        fname_column="file_name",
        use_bboxes=args.use_bboxes,
    )
    dataloader_config = SimpleNamespace(
        batch_size=args.bs,
        n_workers=args.n_jobs,
        pin_memory=True,
        sampler=SimpleNamespace(type="default"),
    )
    return get_loader(
        args.dataset_path,
        df,
        transform=transform_test,
        dataset_config=dataset_config,
        dataloader_config=dataloader_config,
        split="test",
    )


def get_embedding_size(config, args, model=None):
    if model is not None and config.head.type == "no_head":
        return model.backbone_out_feats
    return int(config.embeddings_size) if "embeddings_size" in config else args.emb_size


def run_torch_model(config, args, data_loader, ids_to_labels, eval_status):
    device = get_device(args.device or config.device)
    model = (
        load_emb_model_and_weights(
            config_backbone=config.backbone,
            config_head=config.head,
            weights=args.weights,
            device=device,
        )
        .to(device)
        .eval()
    )

    emb_size = get_embedding_size(config, args, model=model)
    return generate_embeddings(
        model=model,
        model_type="torch",
        data_loader=data_loader,
        ids_to_labels=ids_to_labels,
        emb_size=emb_size,
        device=device,
        save_path=args.save_path,
        eval_status=eval_status,
    )


def run_exported_model(config, args, data_loader, ids_to_labels, eval_status):
    device = get_device(args.device or config.device)
    model_info = load_model_except_torch(
        args.weights,
        model_type=args.model_type,
        device=device,
    )
    emb_size = get_embedding_size(config, args)
    return generate_embeddings(
        model=model_info["model"],
        model_type=args.model_type,
        data_loader=data_loader,
        ids_to_labels=ids_to_labels,
        emb_size=emb_size,
        device=device,
        save_path=args.save_path,
        eval_status=eval_status,
        model_info=model_info,
    )


def generate_embeddings(
    model,
    model_type,
    data_loader,
    ids_to_labels,
    emb_size,
    device,
    save_path,
    eval_status=None,
    model_info=None,
):
    n_samples = len(data_loader.dataset)
    embeddings = np.zeros((n_samples, emb_size), dtype=np.float32)
    labels = np.zeros(n_samples, dtype=object)
    file_names = np.zeros(n_samples, dtype=object)
    index = 0

    if model_type.startswith("tf"):
        import tensorflow as tf
    input_details = model_info.get("input_details") if model_info else None
    output_details = model_info.get("output_details") if model_info else None

    with torch.no_grad():
        for data, targets, file_nms in tqdm(data_loader, total=len(data_loader)):
            if model_type in ["torch", "traced"]:
                data = data.to(device)
                output = model(data)
            elif model_type == "onnx":
                output = model.run(None, {"input": data.numpy()})[0]
            elif model_type in ["tf_32", "tf_16", "tf_int", "tf_dyn"]:
                data_np = torch.permute(data, (0, 2, 3, 1)).numpy()
                tf_data = tf.convert_to_tensor(data_np)
                output = model(input=tf_data)["output"]
            elif model_type == "tf_full_int":
                data_np = torch.permute(data, (0, 2, 3, 1)).numpy()
                input_scale, input_zero_point = input_details["quantization"]
                output_scale, output_zero_point = output_details["quantization"]
                tf_sample_int8 = data_np / input_scale + input_zero_point
                tf_sample_int8 = tf_sample_int8.astype(input_details["dtype"])
                tf_sample_int8 = tf.convert_to_tensor(tf_sample_int8)
                output = model(input=tf_sample_int8)
                output = output_scale * (
                    output["output"].astype(np.float32) - output_zero_point
                )
            else:
                raise ValueError(f"Unsupported model_type: {model_type}")

            if torch.is_tensor(output):
                output = output.cpu().numpy()

            batch_size = output.shape[0]
            lbls = [ids_to_labels[t] for t in targets.cpu().numpy()]
            embeddings[index : (index + batch_size), :] = output
            labels[index : (index + batch_size)] = lbls
            file_names[index : (index + batch_size)] = file_nms
            index += batch_size

    save_path.mkdir(exist_ok=True, parents=True)
    np.savez(
        save_path / "embeddings.npz",
        embeddings=embeddings,
        labels=labels,
        file_names=file_names,
        eval_status=eval_status,
    )


if __name__ == "__main__":
    args = resolve_args(parse_args())
    config = OmegaConf.load(args.config)
    df, eval_status = read_annotations(args)
    data_loader, dataset = build_dataloader(config, args, df)
    ids_to_labels = dataset.get_ids_to_labels()

    if args.model_type == "torch":
        run_torch_model(config, args, data_loader, ids_to_labels, eval_status)
    else:
        run_exported_model(config, args, data_loader, ids_to_labels, eval_status)
