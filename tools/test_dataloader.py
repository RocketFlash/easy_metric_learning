import sys
sys.path.append("./")

import hydra
import torch
from src.data.utils import (get_train_val_split,
                            get_labels_to_ids,
                            get_object_from_omegaconf)
from src.data import get_loader
from src.transform import get_transform
from src.visualization import save_batch_grid
from tqdm import tqdm


@hydra.main(version_base=None,
            config_path='../configs/',
            config_name='config')
def test_dataloader(config):
    annotations = get_object_from_omegaconf(config.dataset.annotations)
    root_dir = get_object_from_omegaconf(config.dataset.dir)

    df_train, df_valid = get_train_val_split(
        annotation=annotations, 
        fold=config.dataset.fold
    )

    print(df_train)
    print(df_valid)

    labels_to_ids = get_labels_to_ids(config.dataset.labels_to_ids_path)

    transform_train = get_transform(config.transform.train)
    transform_valid = get_transform(config.transform.valid)

    train_loader, train_dataset = get_loader(
        root_dir,
        df_train,
        transform=transform_train,
        dataset_config=config.dataset,
        dataloader_config=config.dataloader,
        labels_to_ids=labels_to_ids,
        split='train'
    )
    labels_to_ids = train_dataset.get_labels_to_ids()
    ids_to_labels = train_dataset.get_ids_to_labels()

    valid_loader, valid_dataset = get_loader(
        root_dir,
        df_valid, 
        transform=transform_valid,
        dataset_config=config.dataset,
        dataloader_config=config.dataloader,
        labels_to_ids=labels_to_ids,
        split='valid'
    )

    train_loader = tqdm(train_loader, total=int(len(train_loader)))
    valid_loader = tqdm(valid_loader, total=int(len(valid_loader)))

    for batch_index, (images, annos) in enumerate(train_loader):
        labels = [ids_to_labels[anno.item()] for anno in annos]
        save_batch_grid(
            images, 
            labels,
            config.backbone.norm_std,
            config.backbone.norm_mean,
            save_dir='./tmp', 
            split='train', 
            batch_index=batch_index
        )
        if batch_index+1>=n_batches: break

    for batch_index, (images, annos) in enumerate(valid_loader):
        labels = [ids_to_labels[anno.item()] for anno in annos]
        save_batch_grid(
            images, 
            labels,
            config.backbone.norm_std,
            config.backbone.norm_mean,
            save_dir='./tmp', 
            split='valid', 
            batch_index=batch_index
        )
        if batch_index+1>=n_batches: break


if __name__ == '__main__':
    test_dataloader()