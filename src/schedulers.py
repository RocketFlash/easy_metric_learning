from torch.optim.lr_scheduler import CyclicLR, StepLR, ReduceLROnPlateau, CosineAnnealingLR, MultiStepLR


def get_scheduler(optimizer, scheduler_config):
    scheduler_type = scheduler_config['SHEDULER_TYPE']
    if scheduler_type == 'cyclic':
        scheduler = CyclicLR(optimizer, base_lr=scheduler_config['BASE_LR'],
                             max_lr=scheduler_config['MAX_LR'])
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=scheduler_config['T_MAX'],
                                      eta_min=float(scheduler_config['ETA_MIN']))
    elif scheduler_type == 'plato':
        scheduler = ReduceLROnPlateau(optimizer, mode=scheduler_config['MODE'],
                                      factor=scheduler_config['FACTOR'],
                                      patience=scheduler_config['PATIENCE'],
                                      verbose=scheduler_config['VERBOSE'])
    else:
        scheduler = MultiStepLR(optimizer,
                                milestones=scheduler_config["STEPS"],
                                gamma=scheduler_config["GAMMA"])
    return scheduler