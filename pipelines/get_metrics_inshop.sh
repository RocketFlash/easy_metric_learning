#!/bin/bash

WORK_DIR_PATH="/home/ubuntu/easy_metric_learning/work_dirs/mixed_dataset_arcface_tf_efficientnetv2_b1.in1k_m_0_5_s_30_fold0/"
DATASET_NAME="inshop"
DATASET_PATH="/datasets/metric_learning/$DATASET_NAME/"
DATASET_CSV="$DATASET_PATH/dataset_info.csv"
BS=500
N_JOBS=30
TOP_N=10

echo "$WORK_DIR_PATH"
python tools/generate_embeddings.py\
    --work_folder "$WORK_DIR_PATH"\
    --dataset_path "$DATASET_PATH"\
    --dataset_csv "$DATASET_CSV"\
    --n_jobs $N_JOBS\
    --bs "$BS"\
    --dataset_type inshop
python tools/nearest_search.py\
    --embeddings "$WORK_DIR_PATH/embeddings/$DATASET_NAME/embeddings.npz"\
    --top_n "$TOP_N"\
    --n_jobs "$N_JOBS"\
    --faiss_gpu
python tools/calculate_metrics.py\
    --nearest_csv "$WORK_DIR_PATH/embeddings/$DATASET_NAME/nearest_top10.feather"



