import sys
sys.path.append("./")

import hydra
from src.loss import get_loss
from src.scheduler import get_scheduler
from src.optimizer import get_optimizer
from src.utils import seed_everything, get_device
from src.data.utils import (get_train_val_split,
                            get_labels_to_ids,
                            get_object_from_omegaconf)
from src.data import get_loader
from src.transform import get_transform
from src.model import get_model


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

    labels_to_ids = get_labels_to_ids(config.dataset.labels_to_ids_path)
    transform_train = get_transform(config.transform.train)

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

    device = get_device(config.device)
    model = get_model(
        config.backbone,
        config.head,
        margin_config=config.margin,
        n_classes=len(labels_to_ids)
    ).to(device)

    loss_fn = get_loss(loss_config=config.loss).to(device)
    optimizer = get_optimizer(model, optimizer_config=config.optimizer)
    scheduler = get_scheduler(optimizer, scheduler_config=config.scheduler)
    print(optimizer)
    print(scheduler)
    
    for batch_index, (images, annos) in enumerate(train_loader):
        images = images.to(device)
        annos  = annos.to(device)

        pred = model(images, annos)
        loss = loss_fn(pred, annos)
        print(loss)
        break


if __name__ == '__main__':
    test_dataloader()