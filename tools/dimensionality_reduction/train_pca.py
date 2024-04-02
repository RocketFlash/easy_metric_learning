import sys
sys.path.append("./")

import torch
import pickle
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import hydra
from pathlib import Path
import multiprocessing
from omegaconf import OmegaConf

from src.logger import Logger
from src.data import get_train_data_from_config
from src.model import get_model
from src.utils import (load_checkpoint,
                       load_model_except_torch,
                       seed_everything, 
                       get_device)
from src.evaluator import get_evaluator


@hydra.main(version_base=None,
            config_path='../../configs/',
            config_name='config_pca')
def test(config):
    seed_everything(config.random_state)

    if config.n_workers=='auto':
        config.n_workers = multiprocessing.cpu_count()

    work_dir = Path(config.work_dirs) / config.run_name
    work_dir.mkdir(exist_ok=True, parents=True)

    logger = Logger(work_dir / "log_test.txt")
    if config.debug: logger.info('DEBUG MODE')
    OmegaConf.save(config, work_dir / "config_test.yaml")
    
    data_info = get_train_data_from_config(config, logger=logger) 
    data_info_train = data_info.train

    embeddings_path = work_dir / f'{data_info_train.dataset_name}_embeddings.npz'   
    device = get_device(config.device)

    if not embeddings_path.is_file():
        if config.model_type=='torch':
            if config.model_config:
                model_config = OmegaConf.load(config.model_config)
                config.backbone = model_config.backbone
                config.head = model_config.head

            model = get_model(
                config_backbone=config.backbone,
                config_head=config.head,
            ).to(device)

            if config.head.type=='no_head':
                config.embeddings_size = model.backbone_out_feats
                logger.info(f'Embeddings size changed to backbone output size {config.embeddings_size}')

            logger.info_model(config)

            if config.weights is not None:
                checkpoint_data = load_checkpoint(
                    config.weights, 
                    model=model, 
                    logger=logger, 
                    mode='emb'
                )
                model = checkpoint_data['model'] if 'model' in checkpoint_data else model

            model_info = {
                'model' : model,
                'model_type' : 'torch'
            }
        else:
            model_info = load_model_except_torch(
                config.weights, 
                model_type=config.model_type, 
                device=device,
                logger=logger
            )
    
        evaluator = get_evaluator(
            config,
            save_dir=work_dir,
            device=device,
            model_info=model_info,
            is_eval=False
        )
        
        logger.info(f'Generate embeddings on {data_info_train.dataset_name}')    
        embeddings, labels, file_names = evaluator.generate_embeddings(
            data_info_train,
            n_batches=config.n_batches
        )

        np.savez(
            work_dir / f'{data_info_train.dataset_name}_embeddings.npz', 
            embeddings=embeddings, 
            labels=labels,
            file_names=file_names
        )
    else:
        data = np.load(embeddings_path, allow_pickle=True)
        embeddings = data['embeddings']
        file_names = data['file_names']
        labels = data['labels']

    print(embeddings.shape)
    pca = PCA(n_components=config.n_components)
    embeddings_new = pca.fit_transform(embeddings)
    print(embeddings_new.shape)

    pca_save_path = work_dir /f'pca_from{config.embeddings_size}_to{config.n_components}.pkl'
    with open(pca_save_path,'wb') as f:
        pickle.dump(pca, f)



if __name__=="__main__":
    test()