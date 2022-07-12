from .simple import MetricDataset
from .mxdataset import MXDataset

from torch.utils.data import DataLoader

from ..transforms import get_transformations
from ..utils import worker_init_fn


def get_loader(df_names=None,
               data_config=None,
               dataset_type='simple',
               root_dir=None,  
               batch_size=8, 
               img_size=170,
               num_thread=4,
               pin=False,
               test=False,
               split='train',
               transform_name='no_aug'):

    if data_config is not None:
        root_dir       = data_config["DIR"]
        dataset_type   = data_config["DATASET_TYPE"]
        batch_size     = data_config["BATCH_SIZE"]
        num_thread     = data_config["WORKERS"]
        img_size       = data_config['IMG_SIZE']
        transform_name = data_config['TRAIN_AUG']

    if test is False:
        transform = get_transformations(transform_name, image_size=(img_size,img_size))
    else:
        transform = get_transformations('test_aug', image_size=(img_size,img_size))
    
    if dataset_type=='mxdataset':
        dataset = MXDataset(root_dir=root_dir, transform=transform)
    else:
        dataset = MetricDataset(root_dir=root_dir,
                                df_names=df_names,      
                                transform=transform)
    
    shuffle = split=='train'
    drop_last = split=='train'
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_thread,
                             pin_memory=pin,
                             drop_last=drop_last,
                             worker_init_fn=worker_init_fn)
    return data_loader, dataset