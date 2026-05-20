# Easy metric learning 

Simple framework for metric learning training. Just set up the configuration file and start training. This framework uses [Hydra](https://hydra.cc/docs/intro/) for configurations management and [Accelerate](https://github.com/huggingface/accelerate) for distributed training.

## Models <a name="models"></a>

### Backbones <a name="backbones"></a>

It's possible to use any model from [timm](https://github.com/huggingface/pytorch-image-models), [openclip](https://github.com/mlfoundations/open_clip) or [unicom](https://github.com/deepglint/unicom) libraries as a backbone

Ready-to-use configs include IResNet-50/100/200, MobileFaceNet, EdgeFace, KP-RPE ViT, AM-RADIO, OpenCLIP MobileCLIP, and timm DINOv2/DINOv3, SigLIP/SigLIP2, EVA-02, AIMv2, Hiera, ConvNeXt V2, and FastViT variants under [configs/backbone](configs/backbone/).

### Margins <a name="margins"></a>
 - softmax [config](configs/margin/softmax.yaml)
 - arcface [paper](https://arxiv.org/abs/1801.07698) [config](configs/margin/arcface.yaml)
 - adacos [paper](https://arxiv.org/abs/1905.00292) [config](configs/margin/adacos.yaml)
 - adaface [paper](https://arxiv.org/abs/2204.00964) [config](configs/margin/adaface.yaml)
 - magface [paper](https://arxiv.org/abs/2103.06627) [config](configs/margin/magface.yaml)
 - combined (arcface + cosface) [config](configs/margin/combined.yaml)
 - cosface [paper](https://arxiv.org/abs/1801.09414) [config](configs/margin/cosface.yaml)
 - circle loss [paper](https://arxiv.org/abs/2002.10857) [config](configs/margin/circle.yaml)
 - partialfc arcface [paper](https://arxiv.org/abs/2203.15565) [config](configs/margin/partialfc_arcface.yaml)
 - distributed partialfc arcface [paper](https://arxiv.org/abs/2203.15565) [config](configs/margin/distributed_partialfc_arcface.yaml)
 - x2-softmax adaptive angular margin [config](configs/margin/x2_softmax.yaml)
 - qamface quality-adaptive margin [config](configs/margin/qamface.yaml)
 - curricularface [paper](https://arxiv.org/abs/2004.00288)
 - elasticface [paper](https://arxiv.org/abs/2109.09416)
 - sphereface [paper](https://arxiv.org/abs/1704.08063) [config](configs/margin/sphereface.yaml)
 - subcenter arcface [paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560715.pdf)

### Losses <a name="losses"></a>
 - cross entropy [config](configs/loss/cross_entropy.yaml)
 - focal loss [config](configs/loss/focal_loss.yaml)
 - soft cross entropy [config](configs/loss/soft_cross_entropy.yaml)
 - pair circle loss [paper](https://arxiv.org/abs/2002.10857) [config](configs/loss/circle.yaml)
 - uniface UCE [paper](https://openaccess.thecvf.com/content/ICCV2023/html/Zhou_UniFace_An_Unified_Framework_for_Face_Recognition_ICCV_2023_paper.html) [config](configs/loss/uniface_uce.yaml)
 - topofr topology regularizer [paper](https://arxiv.org/abs/2410.10587) [config](configs/loss/topofr.yaml)
 - topofr SDE/GUM + topology regularizer [config](configs/loss/topofr_sde.yaml)
 - transface entropy mining [paper](https://openaccess.thecvf.com/content/ICCV2023/html/Dan_TransFace_Calibrating_Transformer_Training_for_Face_Recognition_from_a_Data-Centric_Perspective_ICCV_2023_paper.html) [config](configs/loss/transface_ehsm.yaml)
 - multi-similarity [config](configs/loss/multi_similarity.yaml)
 - batch-hard triplet [config](configs/loss/batch_hard_triplet.yaml)
 - supervised contrastive / NT-Xent [config](configs/loss/supcon.yaml), [config](configs/loss/ntxent.yaml)
 - proxy-anchor / proxy-NCA [config](configs/loss/proxy_anchor.yaml), [config](configs/loss/proxy_nca.yaml)
 - SFace, BoundaryFace, UniTSFace-style losses [config](configs/loss/sface.yaml), [config](configs/loss/boundaryface.yaml), [config](configs/loss/unitsface.yaml)
 
# Advanced training and evaluation

Advanced sampler configs include PK, class-balanced effective-number, hierarchical PK, and hard-negative samplers under [configs/dataloader](configs/dataloader/). Pair/batch losses should normally be trained with `dataloader=pk` or another sampler that guarantees positive pairs. Cross-batch memory for embedding losses can be enabled with `train.trainer.xbm.enabled=True`; under DDP each rank maintains its own local memory queue, so it is not a synchronized cross-rank memory bank. Hard-negative caches can be refreshed during training with `train.trainer.hard_negative_cache.enabled=True`.

Optimizer configs include AdamW, Sophia-G, LAMB, Muon, and schedule-free AdamW under [configs/optimizer](configs/optimizer/). EMA/SWA averaging, FSDP, and `torch.compile` are configured in [configs/train/trainer/default.yaml](configs/train/trainer/default.yaml). Averaged checkpoints are saved as `last_averaged.pt`, `best_averaged.pt`, `last_averaged_emb.pt`, and `best_averaged_emb.pt` when model averaging is enabled.

Evaluation supports retrieval (`R@K`, `mAP@R`, `NMI`), k-reciprocal re-ranking via `evaluation.knn.rerank.enabled=True`, face verification (`TAR@FAR`) with CSV/LFW/AgeDB/CFP-style pair protocols, IJB template verification, FAISS flat/IVF/HNSW indexes, and MegaFace-style identification metrics (`CMC@K`, `TPIR@FPIR`). Re-ranking computes dense pairwise distances, so memory grows as `O(N^2)` and it should stay disabled for million-scale galleries such as MegaFace distractor sets. See [configs/evaluation/evaluator](configs/evaluation/evaluator/) and [configs/evaluation/knn](configs/evaluation/knn/). Saved embedding archives can also be benchmarked offline with `tools/benchmark_face_verification.py`, `tools/benchmark_ijb.py`, and `tools/benchmark_megaface.py`.

AM-RADIO is loaded through NVIDIA's TorchHub entrypoint and requires network/cache access to the RADIO repository and pretrained weights. MobileFaceNet is implemented locally, but pretrained weights are not bundled; set `checkpoint_path` in [configs/backbone/mobilefacenet.yaml](configs/backbone/mobilefacenet.yaml) to use external weights.

Experiment tracking supports W&B, MLflow, TensorBoard, Aim, and Neptune. Optional trackers are skipped cleanly when their package or required server/project configuration is unavailable.

Face benchmark protocol files can be normalized from raw LFW/AgeDB, CFP-style, IJB, and MegaFace layouts with `python data/face_benchmarks.py ...`. The command writes `dataset_info.csv` plus normalized pair/template CSVs for the evaluator and offline benchmark tools.

Config composition can be checked without launching training:

```bash
python tools/validate_config.py use_wandb=False use_mlflow=False
```

# Install requirements

## Conda environment
Easiest way to work with this repo is to install conda environment from file. First you need to install conda. And then run the following:

```bash
conda env create -f environment.yml
```

This creates a Python 3.10 environment and installs the project with its test
extra via pip. For CUDA-specific PyTorch builds, install the appropriate torch
wheel for your platform before or after creating the environment.

## Manually

If you want to install all required libraries without conda you can instal them manually. First install [pytorch](https://pytorch.org/get-started/locally/). Next it's possible to install all required libraries using requirements.txt file. Just run:

```bash
pip install -r requirements.txt
```

# Prepare dataset

## Open source datasets

To download and prepare one of the following open source datasets:
- [Cars196](http://ftp.cs.stanford.edu/cs/cvgl/CARS196.zip)
- [Stanford Online Products](http://ftp.cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip)
- [CUB_200_2011](https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz)
- [DeepFashion (Inshop)](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html)
- [Aliproducts2020](https://tianchi.aliyun.com/competition/entrance/231780/introduction)
- [RP2K](https://www.pinlandata.com/rp2k_dataset/)
- [Products10K](https://products-10k.github.io/)
- [MET](https://cmp.felk.cvut.cz/met/)
- [H&M](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)
- [LargeFineFoodAI](https://www.kaggle.com/competitions/largefinefoodai-iccv-recognition/overview/evaluation)
- [Shopee](https://www.kaggle.com/competitions/shopee-product-matching)
- [Inaturalist2021](https://github.com/visipedia/inat_comp/tree/master/2021)

You can just run:

```bash
python data/prepare_dataset.py --dataset {dataset_name} --save_path {dataset save path}
```

*--dataset* - should be one of ['sop', 'cars', 'cub', 'inshop', 'aliproducts', 'rp2k', 'products10k', 'met', 'hnm', 'finefood', 'shopee', 'inaturalist_2021']

## Custom dataset
Easiest way to prepare custom dataset is to orginize dataset into the following structure:

```
dataset_folder
│   
│   └───class_1
│       │   image1.jpg
│       │   image2.jpg
│       │   ...
│   └───class_2
│       |   image1.jpg
│       │   image2.jpg
│       │   ...
│
│      ...
│
│   └───class_N
│       │   ...
```

After that you can just run:

```bash
python data/generate_dataset_info.py --dataset_path {path to the dataset_folder}
```

Optional arguments:

*--hashes* - generate image hashes to use them for duplicates filtering

<details>
<summary><b>(Optional) Dataset filtering</b></summary>

If you want to remove duplicates from your dataset you need to generate **dataset_info.csv** file from previous step with *--hashes* argument, next run:

```bash 
python data/filter_dataset_info.py --dataset_info {path to the dataset_info.csv file} --dedup
```

The script will generate **dataset_info_filtered.csv** file which you can use in next steps.

Optional arguments:

*--min_size* - minimal image size. Removes too small images. If image size is less than *min_size* will remove it from filtered dataset_info file.

*--max_size* - maximal image size. Removes too large images. If image size is more than *max_size* will remove it from filtered dataset_info file.

*--threshold* - threshold for duplicates search indicating the maximum amount of hamming distance that can exist between the key image and a candidate image so that the candidate image can be considered as a duplicate of the key image. Should be an int between 0 and 64. Default value is 10.

</details>

<details>
<summary><b>(Optional) K-fold split</b></summary>
If you want to make stratified k-fold spit on custom dataset you can run:

```bash
python data/get_kfold_split.py --dataset_info {path to the dataset_info.csv file}
```

The script will generate **folds.csv** file with 'fold' column. It will also generate **folds_train_only.csv** and **folds_test_only.csv** to use dataset only for training or only for testing.

Optional arguments:

*--k* - number of folds (default: 5)

*--random_seed* - random seed for reproducibility

*--save_name* - save file name (default: folds)
</details>


<details>
<summary><b>(Optional) Split dataset on train and test</b></summary>

If you want to use part of the classes for testing and rest for training just run:

```bash
python data/get_kfold_split.py --dataset_info {path to the dataset_info.csv file} --split_type {name of split type}
```

There are several ways to split dataset on train and test:

1. based on min and max number of samples (**split_type** = minmax). Classes with number of samples in range **[min_n_samples, max_n_samples]** will be used for training and rest for testing
2. based on proportion and frequency (**split_type** = freq)

Optional script arguments:

*--min_n_samples* - min number of samples to select class for training (used when **split_type** == minmax, default: 3)

*--max_n_samples* - max number of samples to select class for training (used when **split_type** == minmax, default: 50)

*--test_ratio* - test classes ratio (used when **split_type** == freq, default: 0.1)

*--min_freq* - min number of samples in frequency bin to split bin, if less will add whole bin in training set (used when **split_type** == freq, default: 10)

*--random_seed* - random seed for reproducibility

</details>

# Training

## Simple example

In the **configs** folder you will find main training configuration file **config_train.yaml**. It contains default training parameters. For example default backbone set to *efficientnetv2_b1* and default dataset is *cars196*. If you want to change any of the parameters you can enter it as an argument. Let's say you want to train a model for 10 epochs with backbone *openclip_vit_b32.yaml* , margin *arcface* on dataset *inshop* and evaluate model on *products10k* with batch size 32. You can change the values in the configuration file or set what you want as an argument and the remaining parameters will be default parameters from **config_train.yaml**. You can use the following command:

```bash
python tools/train.py backbone=openclip_vit_b32 dataset=inshop evaluation/data=products10k batch_size=32 epochs=10
```

That's all, the result will be saved in the folder **work_dirs**. 

Configuration files are hierarchical, so you can create and configure separate configurations for individual modules. For example, you can create a new configuration file for a new backbone or for a new loss function. All configurations can be found in the **configs** folder. Feel free to modify existing configurations or create new ones.


## Distributed training

If you want to use MultiGPU, MultiCPU, MultiNPU or TPU training or if you want to use several machines (nodes), you need to set up [Accelerate](https://github.com/huggingface/accelerate) framework. Just run:

```bash
accelerate config
```

All Accelerate lirary configuration parameters could be found [here](https://huggingface.co/docs/accelerate/index).

Once you have configured Accelerate library you can run distributed training using the following command:

```bash
accelerate launch tools/train.py backbone=openclip_vit_l {and other configuration parameters}
```
