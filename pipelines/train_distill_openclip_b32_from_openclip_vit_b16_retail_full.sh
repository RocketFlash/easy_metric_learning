#!/bin/bash

accelerate launch tools/train.py\
    backbone=openclip_vit_b32\
    distillation/teacher=arcface_openclip_vit_b16_retail_full.yaml\
    distillation/trainer=distill_loss_only_l2\
    batch_size=64\
    visualize_batch=False\
    ddp=True\
    dataset=retail_full\
    evaluation/data=retail_open_source\


