#!/bin/bash

accelerate launch tools/train.py\
    backbone=efficientnetv2_b3\
    distillation/teacher=arcface_openclip_vit_b16_retail_full.yaml\
    batch_size=256\
    visualize_batch=False\
    ddp=True\
    dataset=retail_full\
    evaluation/data=retail_open_source\
    backbone_lr_scaler=1\
    grad_accum_steps=1\
    use_wandb=False\
    use_mlflow=False\


