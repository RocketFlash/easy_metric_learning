#!/bin/bash

WORK_DIRS_PATH="/home/ubuntu/easy_metric_learning/work_dirs/"
DATASET_NAME="product_recognition"
DATASET_PATH="/datasets/metric_learning/$DATASET_NAME/"
DATASET_CSV="$DATASET_PATH/dataset_test.csv"
REF_CSV="$DATASET_PATH/ref.csv"
TEST_CSV="$DATASET_PATH/ref_test.csv"
BS=500
N_JOBS=30
TOP_N=10

for d in $WORK_DIRS_PATH/*; do
  if [ -d "$d" ]; then
    echo "$d"
    python tools/generate_embeddings.py\
        --work_folder "$d"\
        --dataset_path "$DATASET_PATH"\
        --dataset_csv "$DATASET_CSV"\
        --bs "$BS"
    python tools/nearest_search.py\
        --embeddings "$d/embeddings/$DATASET_NAME/embeddings.npz"\
        --ref_csv "$REF_CSV"\
        --test_csv "$TEST_CSV"\
        --top_n "$TOP_N"\
        --n_jobs "$N_JOBS"
    python tools/calculate_metrics.py\
        --nearest_csv "$d/embeddings/$DATASET_NAME/nearest_top10.feather"
  fi
done


