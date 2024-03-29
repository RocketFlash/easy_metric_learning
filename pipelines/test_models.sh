#!/bin/bash

DATASET_NAME="cars196"

# #########################################
# # Efficientnetv2 b1 models
# #########################################

echo "test untrained efficientnetv2_b1 on $DATASET_NAME datasets"
python tools/test.py\
        backbone=efficientnetv2_b1\
        head=base\
        evaluation/data=$DATASET_NAME\
        embeddings_size=512\
        img_h=224\
        img_w=224\
        run_name="$DATASET_NAME"_efficientnetv2_b1_img224x224_emb512_untrained


echo "test old trained efficientnetv2_b1 on $DATASET_NAME datasets"
python tools/test.py\
        model_type=traced\
        evaluation/data=$DATASET_NAME\
        weights=/home/ubuntu/trained_weights/metric/arcface_efnet_b1_traced.pt\
        embeddings_size=512\
        img_h=224\
        img_w=224\
        run_name="$DATASET_NAME"_efficientnetv2_b1_img224x224_emb512_old

echo "test old trained distilled efficientnetv2_b1 on $DATASET_NAME datasets"
python tools/test.py\
        model_type=traced\
        evaluation/data=$DATASET_NAME\
        weights=/home/ubuntu/trained_weights/metric/distill_arcface_tf_efficientnetv2_b1.in1k_im224_emb512_traced.pt\
        embeddings_size=512\
        img_h=224\
        img_w=224\
        run_name="$DATASET_NAME"_distill_efficientnetv2_b1_img224x224_emb512_trained_old

# #########################################
# # Efficientnetv2 b3 models
# #########################################

echo "test untrained efficientnetv2_b3 on $DATASET_NAME datasets"
python tools/test.py\
        backbone=efficientnetv2_b3\
        head=base\
        evaluation/data=$DATASET_NAME\
        embeddings_size=512\
        img_h=224\
        img_w=224\
        run_name="$DATASET_NAME"_efficientnetv2_b3_img224x224_emb512_untrained


# #########################################
# # Unicom b32 models
# #########################################

echo "test untrained unicom vit b32 without head on $DATASET_NAME datasets"
python tools/test.py\
        backbone=unicom_vit_b32\
        head=no_head\
        evaluation/data=$DATASET_NAME\
        img_h=224\
        img_w=224\
        run_name="$DATASET_NAME"_unicom_vit_b32_img224x224_emb512_untrained_no_head

        
echo "test untrained unicom vit b32 with head on $DATASET_NAME datasets"
python tools/test.py\
        backbone=unicom_vit_b32\
        head=base\
        embeddings_size=512\
        evaluation/data=$DATASET_NAME\
        img_h=224\
        img_w=224\
        run_name="$DATASET_NAME"_unicom_vit_b32_img224x224_emb512_untrained


echo "test trained unicom_vit_b32 on $DATASET_NAME datasets"
python tools/test.py\
        backbone=unicom_vit_b32\
        head=no_head\
        model_type=torch\
        evaluation/data=$DATASET_NAME\
        weights=work_dirs/train/arcface_unicom_vit_b32_no_head_mixed1_img224x224_emb512/last_emb.pt\
        embeddings_size=512\
        img_h=224\
        img_w=224\
        run_name="$DATASET_NAME"_unicom_vit_b32_img224x224_emb512_trained_on_mixed1


# #########################################
# # Unicom b16 models
# #########################################

echo "test untrained unicom vit b16 without head on $DATASET_NAME datasets"
python tools/test.py\
        backbone=unicom_vit_b16\
        head=no_head\
        evaluation/data=$DATASET_NAME\
        img_h=224\
        img_w=224\
        run_name="$DATASET_NAME"_unicom_vit_b16_img224x224_emb768_untrained_no_head


echo "test untrained unicom vit b16 with head on $DATASET_NAME datasets"
python tools/test.py\
        backbone=unicom_vit_b16\
        head=base\
        embeddings_size=512\
        evaluation/data=$DATASET_NAME\
        img_h=224\
        img_w=224\
        run_name="$DATASET_NAME"_unicom_vit_b16_img224x224_emb512_untrained


# #########################################
# # Unicom l14 models
# #########################################

echo "test untrained unicom vit l14 without head on $DATASET_NAME datasets"
python tools/test.py\
        backbone=unicom_vit_l14\
        head=no_head\
        evaluation/data=$DATASET_NAME\
        img_h=224\
        img_w=224\
        run_name="$DATASET_NAME"_unicom_vit_l14_img224x224_emb_768_untrained_no_head


echo "test untrained unicom vit l14 with head on $DATASET_NAME datasets"
python tools/test.py\
        backbone=unicom_vit_l14\
        head=base\
        embeddings_size=512\
        evaluation/data=$DATASET_NAME\
        img_h=224\
        img_w=224\
        run_name="$DATASET_NAME"_unicom_vit_l14_img224x224_emb512_untrained


# #########################################
# # Unicom l14_336 models
# #########################################

echo "test untrained unicom vit l14 336 without head on $DATASET_NAME datasets"
python tools/test.py\
        backbone=unicom_vit_l14_336\
        head=no_head\
        evaluation/data=$DATASET_NAME\
        batch_size=64\
        img_h=336\
        img_w=336\
        run_name="$DATASET_NAME"_unicom_vit_l14_img336x336_emb_768_untrained_no_head


echo "test untrained unicom vit l14 with head on $DATASET_NAME datasets"
python tools/test.py\
        backbone=unicom_vit_l14_336\
        head=base\
        embeddings_size=512\
        evaluation/data=$DATASET_NAME\
        batch_size=64\
        img_h=336\
        img_w=336\
        run_name="$DATASET_NAME"_unicom_vit_l14_img336x336_emb512_untrained


# #########################################
# # OpenCLIP b32 models
# #########################################

echo "test untrained openclip vit b32 without head on $DATASET_NAME datasets"
python tools/test.py\
        backbone=openclip_vit_b32\
        head=no_head\
        evaluation/data=$DATASET_NAME\
        img_h=224\
        img_w=224\
        run_name="$DATASET_NAME"_openclip_vit_b32_img224x224_emb512_untrained_no_head

        
echo "test untrained openclip vit b32 with head on $DATASET_NAME datasets"
python tools/test.py\
        backbone=openclip_vit_b32\
        head=base\
        embeddings_size=512\
        evaluation/data=$DATASET_NAME\
        img_h=224\
        img_w=224\
        run_name="$DATASET_NAME"_openclip_vit_b32_img224x224_emb512_untrained


echo "test trained openclip_vit_b32 on $DATASET_NAME datasets"
python tools/test.py\
        backbone=openclip_vit_b32\
        head=base\
        model_type=torch\
        evaluation/data=$DATASET_NAME\
        weights=work_dirs/train/arcface_openclip-ViT-B32_laion2b_base_mixed2_img224x224_emb512/last_emb.pt\
        embeddings_size=512\
        img_h=224\
        img_w=224\
        run_name="$DATASET_NAME"_openclip_vit_b32_img224x224_emb512_trained_on_mixed2

# #########################################
# # OpenCLIP b16 models
# #########################################

echo "test untrained openclip vit b16 without head on $DATASET_NAME datasets"
python tools/test.py\
        backbone=openclip_vit_b16\
        head=no_head\
        evaluation/data=$DATASET_NAME\
        img_h=224\
        img_w=224\
        run_name="$DATASET_NAME"_openclip_vit_b16_img224x224_emb512_untrained_no_head

        
echo "test untrained openclip vit b16 with head on $DATASET_NAME datasets"
python tools/test.py\
        backbone=openclip_vit_b16\
        head=base\
        embeddings_size=512\
        evaluation/data=$DATASET_NAME\
        img_h=224\
        img_w=224\
        run_name="$DATASET_NAME"_openclip_vit_b16_img224x224_emb512_untrained


# #########################################
# # OpenCLIP l14 models
# #########################################

echo "test untrained openclip vit l14 without head on $DATASET_NAME datasets"
python tools/test.py\
        backbone=openclip_vit_l14\
        head=no_head\
        evaluation/data=$DATASET_NAME\
        img_h=224\
        img_w=224\
        run_name="$DATASET_NAME"_openclip_vit_l14_img224x224_emb_768_untrained_no_head


echo "test untrained openclip vit l14 with head on $DATASET_NAME datasets"
python tools/test.py\
        backbone=openclip_vit_l14\
        head=base\
        embeddings_size=512\
        evaluation/data=$DATASET_NAME\
        img_h=224\
        img_w=224\
        run_name="$DATASET_NAME"_openclip_vit_l14_img224x224_emb512_untrained


echo "test trained openclip_vit_l14 on $DATASET_NAME datasets"
python tools/test.py\
        backbone=openclip_vit_l14\
        head=base\
        model_type=torch\
        evaluation/data=$DATASET_NAME\
        weights=work_dirs/train/arcface_openclip-ViT-L_laion2b_base_mixed2_img224x224_emb512/last_emb.pt\
        embeddings_size=512\
        img_h=224\
        img_w=224\
        run_name="$DATASET_NAME"_openclip_vit_l14_img224x224_emb512_trained_on_mixed2

# #########################################
# # OpenCLIP ConvNext models
# #########################################

echo "test old trained openclip_convnext on $DATASET_NAME datasets"
python tools/test.py\
        model_type=traced\
        evaluation/data=$DATASET_NAME\
        weights=/home/ubuntu/trained_weights/metric/arcface_openclip-ConvNext-Base_im224_emb512_traced.pt\
        embeddings_size=512\
        img_h=224\
        img_w=224\
        run_name="$DATASET_NAME"_convnext_img224x224_emb512_trained_old


# #########################################
# # models trained on H100
# #########################################


echo "test trained unicom_vit_b16 on $DATASET_NAME datasets"
python tools/test.py\
        backbone=unicom_vit_b16\
        head=base\
        model_type=torch\
        evaluation/data=$DATASET_NAME\
        weights=work_dirs/train/arcface_unicom_vit_b16_base_retail_open_source_img224x224_emb512_h100/last_emb.pt\
        embeddings_size=512\
        img_h=224\
        img_w=224\
        run_name="$DATASET_NAME"_unicom_vit_b16_img224x224_emb512_trained_on_retail_opensource


echo "test trained unicom_vit_b32 on $DATASET_NAME datasets"
python tools/test.py\
        backbone=unicom_vit_b32\
        head=no_head\
        model_type=torch\
        evaluation/data=$DATASET_NAME\
        weights=work_dirs/train/arcface_unicom_vit_b32_no_head_retail_open_source_img224x224_emb512_h100/last_emb.pt\
        embeddings_size=512\
        img_h=224\
        img_w=224\
        run_name="$DATASET_NAME"_unicom_vit_b32_img224x224_emb512_trained_on_retail_opensource