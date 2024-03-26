#!/bin/bash
SAVE_PATH="/datasets/metric_learning/datasets_opensource"

echo "Prepare cars196 dataset"
python data/prepare_dataset.py\
    --dataset cars\
    --save_path $SAVE_PATH

echo "Prepare CUB200_2011 dataset"
python data/prepare_dataset.py\
    --dataset cub\
    --save_path $SAVE_PATH

echo "Prepare inshop dataset"
python data/prepare_dataset.py\
    --dataset inshop\
    --save_path $SAVE_PATH

echo "Prepare aliproducts dataset"
python data/prepare_dataset.py\
    --dataset aliproducts\
    --save_path $SAVE_PATH

echo "Prepare MET dataset"
python data/prepare_dataset.py\
    --dataset met\
    --save_path $SAVE_PATH

echo "Prepare products10k dataset"
python data/prepare_dataset.py\
    --dataset products10k\
    --save_path $SAVE_PATH

echo "Prepare shopee dataset"
python data/prepare_dataset.py\
    --dataset shopee\
    --save_path $SAVE_PATH

echo "Prepare stanford online products dataset"
python data/prepare_dataset.py\
    --dataset sop\
    --save_path $SAVE_PATH

echo "Prepare R2PK dataset"
python data/prepare_dataset.py\
    --dataset rp2k\
    --save_path $SAVE_PATH

echo "Prepare large fine food dataset"
python data/prepare_dataset.py\
    --dataset finefood\
    --save_path $SAVE_PATH

echo "Prepare inaturalist 2021 dataset"
python data/prepare_dataset.py\
    --dataset inaturalist_2021\
    --save_path $SAVE_PATH






