from .base import BaseEvaluator
from .ddp import DDPEvaluator


def get_evaluator(
        config,
        model,
        save_dir='./',
        device='cpu',
        accelerator=None,
    ):
    if accelerator is not None:
        evaluator = DDPEvaluator(
            config,
            model=model, 
            save_dir=save_dir, 
            accelerator=accelerator,
        )
    else:
        evaluator = BaseEvaluator(
            config,
            model=model, 
            save_dir=save_dir, 
            device=device, 
        )

    return evaluator