from .base import BaseTrainer
from .ddp import DDPTrainer

def get_trainer(
        config,
        model,
        optimizer,
        device,
        accelerator,
        epoch,
        work_dir,
        ids_to_labels=None
    ):

    if config.train.trainer.type == 'distill':
        trainer = None
    else:
        if accelerator is not None:
            trainer = DDPTrainer(
                config,
                model=model,
                optimizer=optimizer, 
                accelerator=accelerator,
                epoch=epoch,
                work_dir=work_dir,
                ids_to_labels=ids_to_labels
            )
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