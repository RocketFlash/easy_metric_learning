import sys

sys.path.append("./")

import hydra
from src.loss import get_loss
from src.scheduler import get_scheduler, get_warmup_scheduler
from src.optimizer import get_optimizer
from src.model import get_model
from src.utils import get_device
from src.data import get_train_data_from_config
from src.trainer.loss_inputs import calculate_weighted_loss


@hydra.main(version_base=None, config_path="../configs/", config_name="config_train")
def test_dataloader(config):
    data_info = get_train_data_from_config(config)
    train_loader = data_info.train.dataloader
    labels_to_ids = data_info.train.labels_to_ids
    config.n_classes = data_info.train.dataset_stats.n_classes

    config.margin.id_counts = data_info.train.dataset_stats.id_counts

    device = get_device(config.device)
    model = get_model(
        config_backbone=config.backbone,
        config_head=config.head,
        config_margin=config.margin,
        n_classes=len(labels_to_ids),
    ).to(device)

    loss_fns = get_loss(loss_config=config.loss, device=device)

    optimizer = get_optimizer(model, optimizer_config=config.optimizer)
    scheduler = get_scheduler(optimizer, scheduler_config=config.scheduler)
    warmup_scheduler = get_warmup_scheduler(
        optimizer, scheduler_config=config.scheduler
    )
    print(optimizer)
    print(scheduler)
    print(warmup_scheduler)

    for batch_index, (images, annos, file_names) in enumerate(train_loader):
        images = images.to(device)
        annos = annos.to(device)

        pred, emb = model(images, annos)
        for loss_name, loss_params in loss_fns.items():
            loss = calculate_weighted_loss(
                loss_params,
                pred,
                emb,
                annos,
                images=images,
            )
            print(f"{loss_name} : {loss}")
        break


if __name__ == "__main__":
    test_dataloader()
