#!/bin/bash

accelerate launch tools/train.py\
    backbone=unicom_vit_b32\
    margin=arcface_incremental_m\
    head=no_head\
    train/best_model_criterion=best_R1_on_rp2k\
    dataset=retail_open_source\
    evaluation/data=retail\
    batch_size=256\
    use_mlflow=False\
    load_mode=resume\
    load_checkpoint=work_dirs/train/arcface_unicom_vit_b32_no_head_retail_open_source_img224x224_emb512/last.pt\
    ddp=False



