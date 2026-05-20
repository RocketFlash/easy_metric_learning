import hydra
import torch


def get_scheduler(optimizer, scheduler_config):
    scheduler = hydra.utils.instantiate(
        scheduler_config.get("scheduler"), optimizer=optimizer, _convert_="object"
    )

    return scheduler


def set_scheduler_tmax(scheduler_config, t_max):
    try:
        has_tmax = "T_max" in scheduler_config
    except TypeError:
        has_tmax = hasattr(scheduler_config, "T_max")
    if not has_tmax:
        return

    try:
        scheduler_config.T_max = t_max
    except AttributeError:
        scheduler_config["T_max"] = t_max


def scheduler_steps_per_batch(scheduler):
    scheduler = getattr(scheduler, "scheduler", scheduler)
    return isinstance(
        scheduler,
        (
            torch.optim.lr_scheduler.CyclicLR,
            torch.optim.lr_scheduler.OneCycleLR,
        ),
    )


def scheduler_requires_metric(scheduler):
    scheduler = getattr(scheduler, "scheduler", scheduler)
    return isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)


def step_scheduler(scheduler, metric=None):
    if scheduler_requires_metric(scheduler):
        if metric is None:
            return
        scheduler.step(metric)
    else:
        scheduler.step()


def _has_config_key(config, key):
    try:
        return key in config
    except TypeError:
        return hasattr(config, key)


def get_warmup_scheduler(optimizer, scheduler_config):
    warmup_config = None
    if _has_config_key(scheduler_config, "warmup_scheduler"):
        warmup_config = scheduler_config.warmup_scheduler

    if warmup_config is not None:
        import pytorch_warmup as warmup

        if "adam" in warmup_config.optimizer_type:
            warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
        else:
            warmup_scheduler = warmup.LinearWarmup(
                optimizer, warmup_period=warmup_config.warmup_period
            )
    else:
        warmup_scheduler = None

    return warmup_scheduler
