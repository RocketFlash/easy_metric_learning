import torch.nn as nn
from .mix import MixCriterion
from .focal import FocalLoss 
from .soft_cross_entropy import SoftCrossEntropyLoss


def get_loss(train_config=None, loss_type='cross_entropy', weight=None, gamma=2, num_classes=2, label_smoothing=0):
    if train_config is not None:
        loss_type   = train_config['LOSS_TYPE']
        gamma       = train_config['FOCAL_GAMMA']
        num_classes = train_config['N_CLASSES']

    if loss_type=='cross_entropy':
        if label_smoothing==0:
            loss_fnc = nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fnc = SoftCrossEntropyLoss(label_smoothing=label_smoothing, weight=weight, num_classes=num_classes, reduction='mean')
    elif loss_type=='focal':
        loss_fnc = FocalLoss(gamma=gamma)

    return loss_fnc