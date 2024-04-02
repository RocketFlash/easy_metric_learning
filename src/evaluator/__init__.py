from .base import BaseEvaluator
from .ddp import DDPEvaluator


def get_evaluator(
        config,
        model=None,
        save_dir='./',
        device='cpu',
        accelerator=None,
        model_info=None,
        is_eval=True,
        pca=None
    ):
    if accelerator is not None:
        evaluator = DDPEvaluator(
            config,
            model=model, 
            save_dir=save_dir, 
            accelerator=accelerator,
            is_eval=is_eval,
            pca=pca
        )
    else:
        evaluator = BaseEvaluator(
            config,
            model=model, 
            save_dir=save_dir, 
            device=device, 
            model_info=model_info,
            is_eval=is_eval,
            pca=pca
        )

    return evaluator