import hydra


def get_scheduler(optimizer, scheduler_config):    
    scheduler = hydra.utils.instantiate(
        scheduler_config.get('scheduler'), 
        optimizer=optimizer,
        _convert_="object"
    )
    
    return scheduler


def get_warmup_scheduler(optimizer, scheduler_config):   
    if 'warmup_scheduler' in scheduler_config:
        import pytorch_warmup as warmup
        if 'adam' in scheduler_config.warmup_scheduler.optimizer_type:
            warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
        else:
            warmup_scheduler = warmup.LinearWarmup(
                optimizer, 
                warmup_period=scheduler_config.warmup_period
            )
    else:
        warmup_scheduler = None
    
    return warmup_scheduler