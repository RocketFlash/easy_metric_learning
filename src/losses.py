import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


class SoftCrossEntropyLoss(nn.NLLLoss):
    def __init__(self, label_smoothing=0, weight=None, num_classes=2, **kwargs):
        super(SoftCrossEntropyLoss, self).__init__(**kwargs)
        self.label_smoothing = label_smoothing
        self.confidence = 1 - self.label_smoothing
        self.num_classes = num_classes
        self.register_buffer('weight', Variable(weight))

        assert label_smoothing >= 0.0 and label_smoothing <= 1.0
    def forward(self, input_data, target):
        input_data = F.log_softmax(input_data, dim=-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(input_data)
            true_dist.fill_(self.label_smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

            if self.weight is not None:
                true_dist.mul_(self.weight)

        return torch.mean(torch.sum(-true_dist * input_data, dim=-1))


def cutmix(batch, alpha):
    data, targets = batch

    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)

    image_h, image_w = data.shape[2:]
    cx = np.random.uniform(0, image_w)
    cy = np.random.uniform(0, image_h)
    w = image_w * np.sqrt(1 - lam)
    h = image_h * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, image_w)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, image_h)))

    data[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]
    targets = (targets, shuffled_targets, lam)

    return data, targets


class CutMixCollator:
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, batch):
        batch = torch.utils.data.dataloader.default_collate(batch)
        batch = cutmix(batch, self.alpha)
        return batch


class CutMixCriterion:
    '''Code from: https://github.com/hysts/pytorch_cutmix'''
    def __init__(self, criterion):
        self.criterion = criterion

    def __call__(self, preds, targets):
        targets1, targets2, lam = targets
        return lam * self.criterion(
            preds, targets1) + (1 - lam) * self.criterion(preds, targets2)


def get_loss(loss_type, weight=None, gamma=2, num_classes=2, label_smoothing=0):
    if loss_type=='cross_entropy':
        if label_smoothing==0:
            loss_fnc = nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fnc = SoftCrossEntropyLoss(label_smoothing=label_smoothing, weight=weight, num_classes=num_classes, reduction='mean')
    elif loss_type=='focal':
        loss_fnc = FocalLoss(gamma=gamma)

    return loss_fnc