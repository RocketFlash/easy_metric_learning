from torch.optim.lr_scheduler import CyclicLR, StepLR, ReduceLROnPlateau, CosineAnnealingLR, MultiStepLR


def get_scheduler(optimizer, optimizer_config):
    scheduler_type = optimizer_config['SHEDULER_TYPE']
    if scheduler_type == 'cyclic':
        scheduler = CyclicLR(optimizer, base_lr=optimizer_config['BASE_LR'],
                             max_lr=optimizer_config['MAX_LR'])
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=optimizer_config['T_MAX'],
                                      eta_min=float(optimizer_config['ETA_MIN']))
    elif scheduler_type == 'plato':
        scheduler = ReduceLROnPlateau(optimizer, mode=optimizer_config['MODE'],
                                      factor=optimizer_config['FACTOR'],
                                      patience=optimizer_config['PATIENCE'],
                                      verbose=optimizer_config['VERBOSE'])
    else:
        scheduler = MultiStepLR(optimizer,
                                milestones=optimizer_config["STEPS"],
                                gamma=optimizer_config["GAMMA"])
    return scheduler