import hydra
import torch


def get_scheduler(optimizer, scheduler_config):
    scheduler = hydra.utils.instantiate(
        scheduler_config.get("scheduler"), optimizer=optimizer, _convert_="object"
    )

    return scheduler


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


def get_warmup_scheduler(optimizer, scheduler_config):
    if "warmup_scheduler" in scheduler_config:
        import pytorch_warmup as warmup

        if "adam" in scheduler_config.warmup_scheduler.optimizer_type:
            warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
        else:
            warmup_scheduler = warmup.LinearWarmup(
                optimizer, warmup_period=scheduler_config.warmup_scheduler.warmup_period
            )
    else:
        warmup_scheduler = None

    return warmup_scheduler
