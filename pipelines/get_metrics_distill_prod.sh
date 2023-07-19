#!/bin/bash

EMBEDDINGS_TEACHER="/home/ubuntu/easy_metric_learning/work_dirs/mixed_dataset_arcface_openclip-ViT-B32_laion2b_m_0_5_s_30_backbone_lr_scaler_fold0/embeddings/product_recognition/embeddings.npz"
EMBEDDINGS_STUDENT="/home/ubuntu/easy_metric_learning/work_dirs/mixed_dataset_arcface_tf_efficientnetv2_b1.in1k_distill_openclip_b32_fold0/embeddings/product_recognition/embeddings.npz"

DATASET_NAME="product_recognition"
DATASET_PATH="/datasets/metric_learning/$DATASET_NAME/"
DATASET_CSV="$DATASET_PATH/dataset_test.csv"
REF_CSV="$DATASET_PATH/ref.csv"
TEST_CSV="$DATASET_PATH/ref_test.csv"
N_JOBS=30
TOP_N=10

python tools/nearest_search.py\
    --embeddings "$EMBEDDINGS_TEACHER"\
    --ref_csv "$REF_CSV"\
    --test_csv "$TEST_CSV"\
    --top_n "$TOP_N"\
    --n_jobs "$N_JOBS"\
    --faiss_gpu\
    --embeddings_test "$EMBEDDINGS_STUDENT"\
    --save_path "results/distill_efnetb1/"
python tools/calculate_metrics.py\
    --nearest_csv "results/distill_efnetb1/nearest_top10.feather"



