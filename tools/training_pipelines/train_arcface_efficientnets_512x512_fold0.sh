echo "training efficientnet-b0"
python ../train.py --config ../configs/arcface_efficientnet_b0.yml
echo "training efficientnet-b1" 
python ../train.py --config ../configs/arcface_efficientnet_b1.yml
echo "training efficientnet-b2" 
python ../train.py --config ../configs/arcface_efficientnet_b2.yml
echo "training efficientnet-b3" 
python ../train.py --config ../configs/arcface_efficientnet_b3.yml 