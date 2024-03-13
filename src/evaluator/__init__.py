from .base import BaseEvaluator
from .faiss import FAISSEvaluator


def get_evaluator(
        config,
        model,
        save_dir='./',
        device='cpu',
    ):
    if config.evaluation.evaluator.type=='faiss':
        evaluator = FAISSEvaluator(
            config,
            model=model, 
            save_dir=save_dir, 
            device=device, 
        )
    else:
        evaluator = BaseEvaluator(
            config,
            model=model, 
            save_dir=save_dir, 
            device=device, 
        )

    return evaluator