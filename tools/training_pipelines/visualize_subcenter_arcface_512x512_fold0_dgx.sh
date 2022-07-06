echo "visualize K=1"
python ../visualize_embeddings.py --save_name k1_train --embeddings work_dirs/subcenter_arcface_efficientnet_b0_fold0_holes_k1/train_embeddings_labels.npy --interactive --plot_3d 
python ../visualize_embeddings.py --save_name k1_valid --embeddings work_dirs/subcenter_arcface_efficientnet_b0_fold0_holes_k1/embeddings_labels.npy --interactive --plot_3d 
echo "visualize K=2"
python ../visualize_embeddings.py --save_name k2_train --embeddings work_dirs/subcenter_arcface_efficientnet_b0_fold0_holes_k2/train_embeddings_labels.npy --interactive --plot_3d 
python ../visualize_embeddings.py --save_name k2_valid --embeddings work_dirs/subcenter_arcface_efficientnet_b0_fold0_holes_k2/embeddings_labels.npy --interactive --plot_3d 
echo "visualize K=3" 
python ../visualize_embeddings.py --save_name k3_train --embeddings work_dirs/subcenter_arcface_efficientnet_b0_fold0_holes_k3/train_embeddings_labels.npy --interactive --plot_3d 
python ../visualize_embeddings.py --save_name k3_valid --embeddings work_dirs/subcenter_arcface_efficientnet_b0_fold0_holes_k3/embeddings_labels.npy --interactive --plot_3d 
echo "visualize K=5" 
python ../visualize_embeddings.py --save_name k5_train --embeddings work_dirs/subcenter_arcface_efficientnet_b0_fold0_holes_k5/train_embeddings_labels.npy --interactive --plot_3d 
python ../visualize_embeddings.py --save_name k5_valid --embeddings work_dirs/subcenter_arcface_efficientnet_b0_fold0_holes_k5/embeddings_labels.npy --interactive --plot_3d 
echo "visualize K=10" 
python ../visualize_embeddings.py --save_name k10_train --embeddings work_dirs/subcenter_arcface_efficientnet_b0_fold0_holes_k10/train_embeddings_labels.npy --interactive --plot_3d 
python ../visualize_embeddings.py --save_name k10_valid --embeddings work_dirs/subcenter_arcface_efficientnet_b0_fold0_holes_k10/embeddings_labels.npy --interactive --plot_3d 