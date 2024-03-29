from .base import BaseEvaluator
from .ddp import DDPEvaluator


def get_evaluator(
        config,
        model=None,
        save_dir='./',
        device='cpu',
        accelerator=None,
        model_info=None
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
            model_info=model_info
        )

    return evaluator