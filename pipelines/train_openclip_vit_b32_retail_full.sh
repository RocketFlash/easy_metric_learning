#!/bin/bash

accelerate launch tools/train.py\
    backbone=openclip_vit_b32\
    margin=arcface_incremental_m\
    head=no_head\
    train/best_model_criterion=best_R1_on_rp2k\
    dataset=retail_full\
    evaluation/data=retail_open_source\
    batch_size=64\
    visualize_batch=False\
    ddp=True


