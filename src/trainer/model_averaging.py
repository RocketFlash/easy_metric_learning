from torch.optim.swa_utils import AveragedModel


def _config_value(config, key, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def get_model_averaging_config(config):
    trainer_config = _config_value(
        _config_value(config, "train", None), "trainer", None
    )
    return _config_value(trainer_config, "model_averaging", None)


def create_ema_avg_fn(decay):
    def ema_avg_fn(averaged_model_parameter, model_parameter, num_averaged):
        return decay * averaged_model_parameter + (1.0 - decay) * model_parameter

    return ema_avg_fn


def create_averaged_model(config, model, device=None):
    averaging_config = get_model_averaging_config(config)
    if not _config_value(averaging_config, "enabled", False):
        return None

    averaging_type = _config_value(averaging_config, "type", "ema")
    if averaging_type == "ema":
        avg_fn = create_ema_avg_fn(_config_value(averaging_config, "decay", 0.999))
    elif averaging_type == "swa":
        avg_fn = None
    else:
        raise ValueError(f"Unknown model_averaging type: {averaging_type}")

    return AveragedModel(model, device=device, avg_fn=avg_fn)


def should_update_averaged_model(config, epoch):
    averaging_config = get_model_averaging_config(config)
    if not _config_value(averaging_config, "enabled", False):
        return False
    return epoch >= _config_value(averaging_config, "start_epoch", 1)


def should_use_averaged_model_for_eval(config):
    averaging_config = get_model_averaging_config(config)
    if not _config_value(averaging_config, "enabled", False):
        return False
    return _config_value(averaging_config, "use_for_eval", True)


def averaged_model_has_updates(averaged_model):
    if averaged_model is None:
        return False
    return int(averaged_model.n_averaged.item()) > 0
