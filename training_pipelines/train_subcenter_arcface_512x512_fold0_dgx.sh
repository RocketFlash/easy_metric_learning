echo "training K=1"
python ../train.py --config ../configs/subcenter_arcface_efficientnet_b0_holes_k1.yml
echo "training K=2"
python ../train.py --config ../configs/subcenter_arcface_efficientnet_b0_holes_k2.yml
echo "training K=3" 
python ../train.py --config ../configs/subcenter_arcface_efficientnet_b0_holes_k3.yml
echo "training K=5" 
python ../train.py --config ../configs/subcenter_arcface_efficientnet_b0_holes_k5.yml
echo "training K=10" 
python ../train.py --config ../configs/subcenter_arcface_efficientnet_b0_holes_k10.yml