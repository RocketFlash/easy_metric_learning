import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import math

class CosFace(nn.Module):
    """
    Implementation of cosface
    Implementation taken from: https://github.com/4uiiurz1/pytorch-adacos 
    """
    def __init__(self, in_features, out_features, ls_eps=0, s=30.0, m=0.35):
        super(CosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ls_eps = ls_eps  # label smoothing
        self.s = s
        self.m = m
        self.W = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, input, label=None):
        x = F.normalize(input)
        W = F.normalize(self.W)

        logits = F.linear(x, W)
        if label is None:
            return logits
       
        target_logits = logits - self.m
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features

        output = logits * (1 - one_hot) + target_logits * one_hot
        output *= self.s

        return output