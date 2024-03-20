import hydra
from easydict import EasyDict as edict


def get_loss(loss_config, device=None, weight=None):
    loss_fns = {}

    for loss_cfg in loss_config.losses:
        loss_fn = hydra.utils.instantiate(
            loss_cfg.get('loss_fn'), 
            weight=weight,
            _convert_="object"
        )
        if device is not None:
            loss_fn = loss_fn.to(device)
            
        loss_fns[loss_cfg.name] = edict({
            'loss_fn': loss_fn, 
            'weight' : loss_cfg.weight
        })
    
    return loss_fns