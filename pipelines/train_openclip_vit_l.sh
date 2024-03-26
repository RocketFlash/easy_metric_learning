#!/bin/bash

accelerate launch tools/train.py\
    backbone=openclip_vit_l\
    margin=arcface_incremental_m\
    train/best_model_criterion=best_R1_on_dataset_v0\
    dataset=mixed2\
    evaluation/data=mixed_opensource_and_dataset_v0\
    batch_size=32\
    ddp=True\
    load_mode=resume\
    load_checkpoint=work_dirs/arcface_openclip-ViT-L_laion2b_base_mixed1_img224x224_emb512/last.pt