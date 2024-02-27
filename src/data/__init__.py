import torch
from torch.utils.data import (Dataset,
                              DataLoader)
from dataclasses import dataclass

from .dataset import get_dataset
from ..sampler import get_sampler
from ..transform import get_transform
from .utils import (worker_init_fn,
                    get_object_from_omegaconf,
                    get_train_val_split,
                    get_labels_to_ids,
                    get_dataset_stats,
                    DatasetStats)


@dataclass
class DataInfo():
    train_loader: DataLoader
    valid_loader: DataLoader
    train_dataset: Dataset
    valid_dataset: Dataset
    dataset_stats: DatasetStats
    labels_to_ids: dict
    ids_to_labels: dict
    n_classes: int
    

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def get_data_from_config(config):
    annotations = get_object_from_omegaconf(config.dataset.annotations)
    root_dir    = get_object_from_omegaconf(config.dataset.dir) 

    df_train, df_valid = get_train_val_split(
        annotation=annotations, 
        fold=config.dataset.fold
    )

    transform_train = get_transform(config.transform.train)
    transform_valid = get_transform(config.transform.valid)

    labels_to_ids   = get_labels_to_ids(config.dataset.labels_to_ids_path)

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

    dataset_stats = get_dataset_stats(
        df_train=df_train, 
        df_valid=df_valid,
        labels_to_ids=labels_to_ids,
        label_column=config.dataset.label_column
    )

    n_classes = len(labels_to_ids)

    return DataInfo(
            train_loader=train_loader,
            valid_loader=valid_loader,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            dataset_stats=dataset_stats,
            labels_to_ids=labels_to_ids,
            ids_to_labels=ids_to_labels,
            n_classes=n_classes
        )


def get_loader(
        root_dir,
        df_names,
        transform,
        dataset_config,
        dataloader_config,
        labels_to_ids=None,
        split='train'
    ):
        
    dataset = get_dataset(
        root_dir, 
        df_names, 
        transform,
        labels_to_ids,
        dataset_config
    )
    
    drop_last = split=='train'
    shuffle   = split=='train'

    sampler = get_sampler(
        labels=dataset.label_ids,
        sampler_config=dataloader_config.sampler
    )
    
    data_loader = DataLoader(
        dataset=dataset,
        sampler=sampler,
        shuffle=shuffle,
        drop_last=drop_last,
        batch_size=dataloader_config.batch_size,
        num_workers=dataloader_config.n_workers,
        pin_memory=dataloader_config.pin_memory,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn
    )
    return data_loader, dataset