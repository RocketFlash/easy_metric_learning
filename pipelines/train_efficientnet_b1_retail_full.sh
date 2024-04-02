#!/bin/bash

accelerate launch tools/train.py\
    batch_size=256\
    visualize_batch=False\
    ddp=True\
    dataset=retail_full\
    evaluation/data=retail_open_source\
    backbone_lr_scaler=1


