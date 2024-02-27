import sys
sys.path.append("./")

import hydra
from src.loss import get_loss
from src.scheduler import (get_scheduler,
                           get_warmup_scheduler)
from src.optimizer import get_optimizer
from src.model import get_model
from src.utils import get_device
from src.data import get_data_from_config



@hydra.main(version_base=None,
            config_path='../configs/',
            config_name='config')
def test_dataloader(config):
    data_info = get_data_from_config(config)
    train_loader = data_info.train_loader
    valid_loader = data_info.valid_loader
    labels_to_ids = data_info.labels_to_ids

    config.margin.id_counts = data_info.dataset_stats.id_counts

    device = get_device(config.device)
    model = get_model(
        config_backbone=config.backbone,
        config_head=config.head,
        config_margin=config.margin,
        n_classes=len(labels_to_ids)
    ).to(device)

    loss_fn = get_loss(loss_config=config.loss).to(device)
    optimizer = get_optimizer(model, optimizer_config=config.optimizer)
    scheduler = get_scheduler(optimizer, scheduler_config=config.scheduler)
    warmup_scheduler = get_warmup_scheduler(optimizer, scheduler_config=config.scheduler)
    print(optimizer)
    print(scheduler)
    print(warmup_scheduler)
    
    for batch_index, (images, annos) in enumerate(train_loader):
        images = images.to(device)
        annos  = annos.to(device)

        pred = model(images, annos)
        loss = loss_fn(pred, annos)
        print(loss)
        break


if __name__ == '__main__':
    test_dataloader()