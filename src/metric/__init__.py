from .gap import GAP

import hydra
from easydict import EasyDict as edict


def get_metric(loss_config, device='cpu', weight=None):
    loss_fns = {}

    for loss_cfg in loss_config.losses:
        loss_fn = hydra.utils.instantiate(
            loss_cfg.get('loss_fn'), 
            weight=weight,
            _convert_="object"
        ).to(device)
        loss_fns[loss_cfg.name] = edict({
            'loss_fn': loss_fn, 
            'weight' : loss_cfg.weight
        })
    
    return loss_fns