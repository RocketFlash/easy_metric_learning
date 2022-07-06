echo "Validation embeddings"
echo "inference K=1"
python ../inference_fold.py --config ../configs/subcenter_arcface_efficientnet_b0_holes_k1.yml
echo "inference K=2"
python ../inference_fold.py --config ../configs/subcenter_arcface_efficientnet_b0_holes_k2.yml
echo "inference K=3" 
python ../inference_fold.py --config ../configs/subcenter_arcface_efficientnet_b0_holes_k3.yml
echo "inference K=5" 
python ../inference_fold.py --config ../configs/subcenter_arcface_efficientnet_b0_holes_k5.yml
echo "inference K=10" 
python ../inference_fold.py --config ../configs/subcenter_arcface_efficientnet_b0_holes_k10.yml

echo "Train embeddings"
echo "inference K=1"
python ../inference_fold.py --config ../configs/subcenter_arcface_efficientnet_b0_holes_k1.yml --train_embeddings
echo "inference K=2"
python ../inference_fold.py --config ../configs/subcenter_arcface_efficientnet_b0_holes_k2.yml --train_embeddings
echo "inference K=3" 
python ../inference_fold.py --config ../configs/subcenter_arcface_efficientnet_b0_holes_k3.yml --train_embeddings
echo "inference K=5" 
python ../inference_fold.py --config ../configs/subcenter_arcface_efficientnet_b0_holes_k5.yml --train_embeddings
echo "inference K=10" 
python ../inference_fold.py --config ../configs/subcenter_arcface_efficientnet_b0_holes_k10.yml --train_embeddings