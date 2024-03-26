#!/bin/bash

accelerate launch tools/train.py\
    backbone=unicom_vit_b32\
    margin=arcface_incremental_m\
    head=no_head\
    train/best_model_criterion=best_R1_on_rp2k\
    dataset=retail\
    evaluation/data=retail\
    batch_size=256\
    ddp=False



