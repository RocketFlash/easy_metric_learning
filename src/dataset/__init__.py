from .simple import MetricDataset
from .mxdataset import MXDataset

import torch
from torch.utils.data import DataLoader

from ..samplers import get_sampler
from ..transform import get_transform
from ..utils import worker_init_fn
from collections import Counter

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def get_loader(df_names=None,
               data_config=None,
               dataset_type='simple',
               data_type='general',
               root_dir=None,  
               batch_size=None, 
               img_size=170,
               num_thread=4,
               pin=True,
               test=False,
               split='train',
               transform_name='no_aug',
               use_cache=False,
               balanced_smplr=True,
               calc_cl_count=False,
               return_filenames=False):

    if data_config is not None:
        root_dir       = data_config["DIR"]
        dataset_type   = data_config["DATASET_TYPE"]
        batch_size     = data_config["BATCH_SIZE"] if batch_size is None else batch_size
        num_thread     = data_config["WORKERS"]
        img_size       = data_config['IMG_SIZE']
        transform_name = data_config['TRAIN_AUG']
        data_type      = data_config["DATA_TYPE"]
        balanced_smplr = data_config["BALANCED_SAMPLER"] if "BALANCED_SAMPLER" in data_config else True
        use_cache      = data_config["USE_CACHE"] if "USE_CACHE" in data_config else False

    if test is False:
        transform = get_transform(transform_name, data_type=data_type, image_size=(img_size,img_size))
    else:
        transform = get_transform('test_aug', data_type=data_type, image_size=(img_size,img_size))
    
    if dataset_type=='mxdataset':
        dataset = MXDataset(root_dir=root_dir, 
                            transform=transform,
                            use_cache=use_cache,
                            calc_cl_count=calc_cl_count)
    else:
        dataset = MetricDataset(root_dir=root_dir,
                                df_names=df_names,      
                                transform=transform,
                                return_filenames=return_filenames)
    
    drop_last = split=='train'
    shuffle = split=='train' and not balanced_smplr

    sampler = None
    if balanced_smplr and split=='train':
        sampler = get_sampler('balanced',
                              labels=dataset.labels,
                              m=1,
                              batch_size=batch_size, 
                              length_before_new_iter=len(dataset.labels))

    data_loader = DataLoader(dataset=dataset,
                             sampler=sampler,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_thread,
                             pin_memory=pin,
                             drop_last=drop_last,
                             worker_init_fn=worker_init_fn,
                             collate_fn=collate_fn)
    return data_loader, dataset