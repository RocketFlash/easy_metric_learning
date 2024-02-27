import sys
sys.path.append("./")

import hydra
from src.data import get_data_from_config
from src.visualization import save_batch_grid
from tqdm import tqdm


@hydra.main(version_base=None,
            config_path='../configs/',
            config_name='config')
def test_dataloader(config):
    n_batches = 10
    data_info = get_data_from_config(config)
    train_loader = data_info.train_loader
    valid_loader = data_info.valid_loader

    train_loader = tqdm(train_loader, total=int(len(train_loader)))
    valid_loader = tqdm(valid_loader, total=int(len(valid_loader)))

    for batch_index, (images, annos) in enumerate(train_loader):
        labels = [data_info.ids_to_labels[anno.item()] for anno in annos]
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
        labels = [data_info.ids_to_labels[anno.item()] for anno in annos]
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