# Easy metric learning 

Simple framework for metric learning training. Just create configuration file and start training.

## Models <a name="models"></a>

### Backbones <a name="backbones"></a>

It's possible to use any model from [timm](https://github.com/huggingface/pytorch-image-models) library as a backbone

### Margins <a name="margins"></a>
 - softmax
 - arcface [paper](https://arxiv.org/abs/1801.07698)
 - adacos [paper](https://arxiv.org/abs/1905.00292)
 - adaface [paper](https://arxiv.org/abs/2204.00964)
 - combined (arcface + cosface)
 - cosface [paper](https://arxiv.org/abs/1801.09414)
 - curricularface [paper](https://arxiv.org/abs/2004.00288)
 - elasticface [paper](https://arxiv.org/abs/2109.09416)
 - sphereface [paper](https://arxiv.org/abs/1704.08063)
 - subcenter arcface [paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560715.pdf)
 

# Install requirements

## Conda environment
Easiest way to work with this repo is to install conda environment from file. First you need to install conda. And then run the following:

```bash
conda env create -f environment.yml
```

This will install a new conda environment with all the required libraries.

## Manually

If you want to install all required libraries without conda you can instal them manually. First install [pytorch](https://pytorch.org/get-started/locally/). Next it's possible to install all required libraries using requirements.txt file. Just run:

```bash
pip install -r requirements.txt
```

# Prepare dataset

Easiest way to work with this repo is to orginize dataset in following structure:

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

## Prepare annotation file
The easiest way to prepare dataset for training is just to run 

```python
python data/prepare_dataset.py --dataset_path {DATASET_PATH}
```

Optional script arguments:

*--dataset_csv* - dataset info csv file generated using step below (See **Optional steps**)

*--k* - K parameter for K-Fold split

*--random_seed* - random seed for K-Fold stratified split

But if you want to filter your dataset (remove too small or too large images or remove duplicates) or if you want to use part of your dataset for testing, you need to generate **dataset_info.csv** file

<details close>

<summary> Optional steps </summary>

## Generate dataset info file (Optional)

First annotation **csv** file should be generated. Generate it using: 

```python
python data/generate_dataset_info.py --dataset_path {DATASET_PATH}
```

Optional script arguments:

*--hashes* - calculate image hashes (use if you want to remove duplicates)

## Filter dataset (Optional)

If you want to remove image duplicates or remove too small or too big images use the following script:

```python
python data/filter_dataset_info.py --dataset_info {DATASET_CSV_FILE} --dedup
```

Optional script arguments:

*--min_size* - minimal image size

*--max_size* - maximal image size

*--treshold* - threshold value for hashes based duplication removal algorithm

## Split dataset on train and test (Optional)

If you want to use part of the classes for testing based on number of samples just use:

```python
python data/get_train_test_sets.py --dataset_csv {DATASET_CSV_FILE}
```

Optional script arguments:

*--min_n_samples* - min number of samples to select class for training (default: 3)

*--max_n_samples* - max number of samples to select class for training (default: 50)

Classes with number of samples in range **[min_n_samples, max_n_samples]** will be used for training and rest for testing

</details>

# Training

First create your configuration yaml file (use [example_config.yaml](https://github.com/RocketFlash/easy_metric_learning/blob/master/configs/example_config.yaml) as template). Next start training using:

```python
python tools/train.py --config {CONFIG FILE}
```

## TODO

- [ ] Add VPL [paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Deng_Variational_Prototype_Learning_for_Deep_Face_Recognition_CVPR_2021_paper.pdf)
- [ ] Add CLIP
- [ ] Add text embeddings