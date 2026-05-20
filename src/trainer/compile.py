def _config_value(config, key, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def get_compile_config(config):
    trainer_config = _config_value(
        _config_value(config, "train", None), "trainer", None
    )
    return _config_value(trainer_config, "compile", None)


def maybe_compile_model(model, config):
    compile_config = get_compile_config(config)
    if not _config_value(compile_config, "enabled", False):
        return model

    import torch

    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available in this PyTorch version")

    return torch.compile(
        model,
        mode=_config_value(compile_config, "mode", None),
        fullgraph=_config_value(compile_config, "fullgraph", False),
        dynamic=_config_value(compile_config, "dynamic", None),
    )
