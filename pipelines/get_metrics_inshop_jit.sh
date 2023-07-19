#!/bin/bash

WEIGHTS_PATH="/home/ubuntu/arcface_efnet_b1_traced.pt"
DATASET_NAME="inshop"
DATASET_PATH="/datasets/metric_learning/$DATASET_NAME/"
DATASET_CSV="$DATASET_PATH/dataset_info.csv"
BS=300
N_JOBS=30
TOP_N=10
IMG_SIZE=170

echo "$WORK_DIR_PATH"
python tools/generate_embeddings.py\
    --weights "$WEIGHTS_PATH"\
    --model_type traced\
    --dataset_path "$DATASET_PATH"\
    --dataset_csv "$DATASET_CSV"\
    --n_jobs $N_JOBS\
    --bs "$BS"\
    --img_size $IMG_SIZE\
    --dataset_type inshop\
    --save_path "results/$DATASET_NAME"
python tools/nearest_search.py\
    --embeddings "results/$DATASET_NAME/embeddings.npz"\
    --top_n "$TOP_N"\
    --n_jobs "$N_JOBS"\
    --faiss_gpu
python tools/calculate_metrics.py\
    --nearest_csv "results/$DATASET_NAME/nearest_top10.feather"



