#!/bin/bash
SAVE_PATH="/datasets/metric_learning/"

echo "Prepare cars196 dataset"
python tools/datasets/prepare_cars196.py\
    --download\
    --save_path $SAVE_PATH

echo "Prepare CUB200_2011 dataset"
python tools/datasets/prepare_cub_200_2011.py\
    --download\
    --save_path $SAVE_PATH

echo "Prepare inshop dataset"
python tools/datasets/prepare_inshop.py\
    --download\
    --save_path $SAVE_PATH

echo "Prepare aliproducts dataset"
python tools/datasets/prepare_aliproducts.py\
    --download\
    --save_path $SAVE_PATH

echo "Prepare MET dataset"
python tools/datasets/prepare_met.py\
    --download\
    --save_path $SAVE_PATH

echo "Prepare products10k dataset"
python tools/datasets/prepare_products10k.py\
    --download\
    --save_path $SAVE_PATH

echo "Prepare shopee dataset"
python tools/datasets/prepare_shopee.py\
    --download\
    --save_path $SAVE_PATH

echo "Prepare stanford online products dataset"
python tools/datasets/prepare_stanford_online_products.py\
    --download\
    --save_path $SAVE_PATH

echo "Prepare R2PK dataset"
python tools/datasets/prepare_r2pk.py\
    --download\
    --save_path $SAVE_PATH






