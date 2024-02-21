import hydra


def get_scheduler(optimizer, scheduler_config):    
    loss_fn = hydra.utils.instantiate(
        scheduler_config.get('scheduler'), 
        optimizer=optimizer,
        _convert_="object"
    )
    
    return loss_fn