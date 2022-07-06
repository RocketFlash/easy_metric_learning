import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


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