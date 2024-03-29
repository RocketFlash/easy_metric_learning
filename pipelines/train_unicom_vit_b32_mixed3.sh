#!/bin/bash

accelerate launch tools/train.py\
    backbone=unicom_vit_b32\
    margin=arcface_incremental_m\
    head=no_head\
    train/best_model_criterion=best_R1_on_rp2k\
    dataset=mixed3\
    evaluation/data=mixed3\
    batch_size=64\
    visualize_batch=False\
    ddp=True


# accelerate launch tools/train.py\
#     backbone=unicom_vit_b32\
#     margin=arcface_incremental_m\
#     head=no_head\
#     dataset=cars196\
#     evaluation/data=cars196\
#     use_wandb=False\
#     use_mlflow=False\
#     batch_size=256\
#     visualize_batch=False\
#     ddp=True



