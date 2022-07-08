from .simple import MetricDataset

from torch.utils.data import DataLoader

from ..transforms import get_transformations
from ..utils import worker_init_fn


def get_loader(df_names,
               data_config=None,
               root_dir=None,  
               batch_size=8, 
               img_size=512,
               num_thread=4,
               pin=False,
               test=False,
               split='train',
               transform_name='no_aug'):

    if data_config is not None:
        root_dir       = data_config["DIR"] 
        batch_size     = data_config["BATCH_SIZE"]
        num_thread     = data_config["WORKERS"]
        img_size       = data_config['IMG_SIZE']
        transform_name = data_config['TRAIN_AUG']

    if test is False:
        transform = get_transformations(transform_name, image_size=(img_size,img_size))
    else:
        transform = get_transformations('test_aug', image_size=(img_size,img_size))
    
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
    return data_loader