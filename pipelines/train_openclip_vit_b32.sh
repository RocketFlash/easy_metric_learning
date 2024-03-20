#!/bin/bash

accelerate launch tools/train.py\
    backbone=openclip_vit_b32\
    margin=arcface_incremental_m\
    train/best_model_criterion=best_R1_on_dataset_v0\
    dataset=mixed2\
    evaluation/data=mixed_opensource_and_dataset_v0\
    batch_size=256\
    ddp=True



