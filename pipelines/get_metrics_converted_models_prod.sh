#!/bin/bash

WORK_DIR_PATH="/home/ubuntu/easy_metric_learning/work_dirs/mixed_dataset_arcface_tf_efficientnetv2_b1.in1k_m_0_5_s_30_fold0/"
DATASET_NAME="product_recognition"
DATASET_PATH="/datasets/metric_learning/$DATASET_NAME/"
DATASET_CSV="$DATASET_PATH/dataset_test.csv"
REF_CSV="$DATASET_PATH/ref.csv"
TEST_CSV="$DATASET_PATH/ref_test.csv"
BS=1
N_JOBS=30
TOP_N=10
MODEL_TYPES=("torch" "traced" "onnx" "tf_32" "tf_16" "tf_dyn" "tf_int" "tf_full_int")

echo "$WORK_DIR_PATH"

for model_type in ${MODEL_TYPES[*]}; do
    echo "Model type: $model_type"
    python tools/generate_embeddings.py\
        --work_folder "$WORK_DIR_PATH"\
        --model_type "$model_type"\
        --dataset_path "$DATASET_PATH"\
        --dataset_csv "$DATASET_CSV"\
        --n_jobs $N_JOBS\
        --bs "$BS"
    python tools/nearest_search.py\
        --embeddings "$WORK_DIR_PATH/embeddings/$DATASET_NAME/embeddings.npz"\
        --ref_csv "$REF_CSV"\
        --test_csv "$TEST_CSV"\
        --top_n "$TOP_N"\
        --n_jobs "$N_JOBS"\
        --faiss_gpu
    python tools/calculate_metrics.py\
        --nearest_csv "$WORK_DIR_PATH/embeddings/$DATASET_NAME/nearest_top10.feather"
done



