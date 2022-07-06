import torch.nn as nn
from .cutmix import CutMixCollator, CutMixCriterion
from .focal import FocalLoss 
from .soft_cross_entropy import SoftCrossEntropyLoss


def get_loss(loss_type, weight=None, gamma=2, num_classes=2, label_smoothing=0):
    if loss_type=='cross_entropy':
        if label_smoothing==0:
            loss_fnc = nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fnc = SoftCrossEntropyLoss(label_smoothing=label_smoothing, weight=weight, num_classes=num_classes, reduction='mean')
    elif loss_type=='focal':
        loss_fnc = FocalLoss(gamma=gamma)

    return loss_fnc