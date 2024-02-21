import torch.nn as nn
import hydra


def get_loss(loss_config, weight=None):

    loss_fn = hydra.utils.instantiate(
        loss_config.get('loss'), 
        weight=weight,
        _convert_="object"
    )
    
    return loss_fn