from ..transform.transface import DynamicPatchAmplitudeMix


def get_dpap_transform(config):
    transform_config = getattr(config, "transform", None)
    dpap_config = getattr(transform_config, "transface_dpap", None)
    if dpap_config is None or not getattr(dpap_config, "enabled", False):
        return None

    return DynamicPatchAmplitudeMix(
        patch_grid=tuple(dpap_config.patch_grid),
        top_k=dpap_config.top_k,
        probability=dpap_config.probability,
        alpha=dpap_config.alpha,
        ratio=dpap_config.ratio,
    )


def apply_dpap_transform(transform, images):
    if transform is None:
        return images
    return transform(images)
