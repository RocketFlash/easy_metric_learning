#!/bin/bash

accelerate launch tools/train.py\
    distillation/teacher=arcface_openclip_vit_b16_retail_full.yaml\
    batch_size=64\
    visualize_batch=False\
    ddp=True\
    dataset=retail_full\
    evaluation/data=retail_open_source\
    backbone_lr_scaler=1


