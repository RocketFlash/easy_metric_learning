from torch.utils.data import (Dataset,
                              DataLoader)
from dataclasses import dataclass

from .dataset import get_dataset
from ..sampler import get_sampler
from ..transform import get_transform
from .utils import (worker_init_fn,
                    get_object_from_omegaconf,
                    get_train_val_split,
                    get_test_split,
                    get_labels_to_ids,
                    get_dataset_stats,
                    collate_fn,
                    combine_dfs,
                    DatasetStats)


@dataclass
class TrainDataInfo():
    dataset_name: str
    train_loader: DataLoader
    valid_loader: DataLoader
    train_dataset: Dataset
    valid_dataset: Dataset
    train_dataset_stats: DatasetStats
    valid_dataset_stats: DatasetStats
    dataset_stats: DatasetStats
    labels_to_ids: dict
    ids_to_labels: dict
    n_classes: int


@dataclass
class TestDataInfo():
    dataset_name: str
    test_loader: DataLoader
    test_dataset: Dataset
    dataset_stats: DatasetStats
    labels_to_ids: dict
    ids_to_labels: dict
    n_classes: int
    

def get_train_data_from_config(config):
    annotations = get_object_from_omegaconf(config.dataset.annotations)
    root_dir    = get_object_from_omegaconf(config.dataset.dir) 

    fold = config.dataset.fold if 'fold' in config.dataset else 0
    df_train, df_valid = get_train_val_split(
        annotation=annotations, 
        fold=fold
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

    train_dataset_stats = get_dataset_stats(
        df=df_train, 
        labels_to_ids=labels_to_ids,
        label_column=config.dataset.label_column,
        split='train'
    )
    
    if df_valid is not None:
        valid_loader, valid_dataset = get_loader(
            root_dir,
            df_valid, 
            transform=transform_valid,
            dataset_config=config.dataset,
            dataloader_config=config.dataloader,
            labels_to_ids=labels_to_ids,
            split='valid'
        )
        valid_dataset_stats = get_dataset_stats(
            df=df_valid, 
            labels_to_ids=labels_to_ids,
            label_column=config.dataset.label_column,
            split='valid'
        )

        df_full = combine_dfs(df_train, df_valid)
        dataset_stats = get_dataset_stats(
            df=df_full, 
            labels_to_ids=labels_to_ids,
            label_column=config.dataset.label_column,
            split='full'
        )
    else:
        valid_loader, valid_dataset = None, None
        valid_dataset_stats = None
        dataset_stats = train_dataset_stats

    n_classes = len(labels_to_ids)

    return TrainDataInfo(
            dataset_name=config.dataset,
            train_loader=train_loader,
            valid_loader=valid_loader,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            train_dataset_stats=train_dataset_stats,
            valid_dataset_stats=valid_dataset_stats,
            dataset_stats=dataset_stats,
            labels_to_ids=labels_to_ids,
            ids_to_labels=ids_to_labels,
            n_classes=n_classes
        )


def get_test_data_from_config(config):
    transform_test = get_transform(config.transform.test)

    data_infos = []
    for dataset_config in config.evaluation.datasets:
        annotations = get_object_from_omegaconf(dataset_config.annotations)
        root_dir    = get_object_from_omegaconf(dataset_config.dir) 

        fold = dataset_config.fold if 'fold' in dataset_config else 0
        df_test = get_test_split(
            annotation=annotations, 
            fold=fold
        )
        labels_to_ids_path = dataset_config.labels_to_ids_path if 'labels_to_ids_path' in dataset_config else None
        labels_to_ids = get_labels_to_ids(labels_to_ids_path)

        test_loader, test_dataset = get_loader(
            root_dir,
            df_test,
            transform=transform_test,
            dataset_config=dataset_config,
            dataloader_config=config.dataloader,
            labels_to_ids=labels_to_ids,
            split='test'
        )
        labels_to_ids = test_dataset.get_labels_to_ids()
        ids_to_labels = test_dataset.get_ids_to_labels()
    
        dataset_stats = get_dataset_stats(
            df=df_test, 
            labels_to_ids=labels_to_ids,
            label_column=dataset_config.label_column
        )

        n_classes = len(labels_to_ids)

        data_info = TestDataInfo(
            dataset_name=dataset_config.name,
            test_loader=test_loader,
            test_dataset=test_dataset,
            dataset_stats=dataset_stats,
            labels_to_ids=labels_to_ids,
            ids_to_labels=ids_to_labels,
            n_classes=n_classes
        )

        data_infos.append(data_info)

    return data_infos


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

    if split=='train':
        sampler = get_sampler(
            labels=dataset.label_ids,
            sampler_config=dataloader_config.sampler
        )
    else:
        sampler = None
    
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