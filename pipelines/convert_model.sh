#!/bin/bash

python tools/train.py\
    backbone=openclip_vit_b32\
    distillation/teacher=arcface_openclip_vit_b16_retail_full.yaml\
    batch_size=64\
    visualize_batch=False\
    ddp=True\
    dataset=retail_full\
    evaluation/data=retail_open_source\


