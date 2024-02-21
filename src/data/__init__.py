import torch
from torch.utils.data import DataLoader

from .dataset import get_dataset
from ..sampler import get_sampler
from .utils import worker_init_fn


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


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