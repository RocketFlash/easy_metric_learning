from .base import BaseEvaluator
from .ddp import DDPEvaluator
from .ijb import DDPIJBTemplateEvaluator, IJBTemplateEvaluator
from .verification import DDPFaceVerificationEvaluator, FaceVerificationEvaluator


def get_evaluator(
    config,
    model=None,
    save_dir="./",
    device="cpu",
    accelerator=None,
    model_info=None,
    is_eval=True,
    pca=None,
):
    evaluator_type = getattr(config.evaluation.evaluator, "type", "base")

    if evaluator_type == "ijb_template" and accelerator is not None:
        evaluator = DDPIJBTemplateEvaluator(
            config,
            model=model,
            save_dir=save_dir,
            device=device,
            accelerator=accelerator,
            is_eval=is_eval,
            pca=pca,
        )
    elif evaluator_type == "ijb_template":
        evaluator = IJBTemplateEvaluator(
            config,
            model=model,
            save_dir=save_dir,
            device=device,
            model_info=model_info,
            is_eval=is_eval,
            pca=pca,
        )
    elif evaluator_type == "face_verification" and accelerator is not None:
        evaluator = DDPFaceVerificationEvaluator(
            config,
            model=model,
            save_dir=save_dir,
            device=device,
            accelerator=accelerator,
            is_eval=is_eval,
            pca=pca,
        )
    elif evaluator_type == "face_verification":
        evaluator = FaceVerificationEvaluator(
            config,
            model=model,
            save_dir=save_dir,
            device=device,
            model_info=model_info,
            is_eval=is_eval,
            pca=pca,
        )
    elif accelerator is not None:
        evaluator = DDPEvaluator(
            config,
            model=model,
            save_dir=save_dir,
            device=device,
            accelerator=accelerator,
            is_eval=is_eval,
            pca=pca,
        )
    else:
        evaluator = BaseEvaluator(
            config,
            model=model,
            save_dir=save_dir,
            device=device,
            model_info=model_info,
            is_eval=is_eval,
            pca=pca,
        )

    return evaluator
