from .base import BaseTrainer


def get_trainer(
        config,
        model,
        optimizer,
        device,
        epoch,
        work_dir,
        ids_to_labels=None
    ):

    if config.train.trainer.type == 'distill':
        trainer = None
    else:
        trainer = BaseTrainer(
            config,
            model=model,
            optimizer=optimizer, 
            device=device,
            epoch=epoch,
            work_dir=work_dir,
            ids_to_labels=ids_to_labels
        )
    
    return trainer