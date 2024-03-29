from .base import BaseTrainer
from .ddp import DDPTrainer
from .distill import DistillTrainer
from ..model import get_model_teacher


def get_trainer(
        config,
        model,
        optimizer,
        device,
        accelerator,
        epoch,
        work_dir,
        ids_to_labels=None,
    ):

    if config.distillation.teacher.model is not None:
        model_teacher = get_model_teacher(config.distillation.teacher)
        if config.ddp:
            model_teacher = accelerator.prepare(model_teacher)
        else:
            model_teacher = model_teacher.to(device)

        trainer = DistillTrainer(
            config,
            model=model,
            model_teacher=model_teacher,
            optimizer=optimizer, 
            device=device,
            accelerator=accelerator,
            epoch=epoch,
            work_dir=work_dir,
            ids_to_labels=ids_to_labels
        )
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