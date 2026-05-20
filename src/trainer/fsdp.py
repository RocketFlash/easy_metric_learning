def _config_value(config, key, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def get_fsdp_config(config):
    trainer_config = _config_value(
        _config_value(config, "train", None), "trainer", None
    )
    return _config_value(trainer_config, "fsdp", None)


def is_fsdp_enabled(config):
    return bool(_config_value(get_fsdp_config(config), "enabled", False))


def maybe_wrap_fsdp(model, config):
    fsdp_config = get_fsdp_config(config)
    if not _config_value(fsdp_config, "enabled", False):
        return model

    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "FSDP requires an initialized torch.distributed process group"
        )

    mixed_precision = None
    if _config_value(fsdp_config, "mixed_precision", False):
        import torch

        mixed_precision = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )

    return FSDP(
        model,
        use_orig_params=_config_value(fsdp_config, "use_orig_params", True),
        mixed_precision=mixed_precision,
    )
