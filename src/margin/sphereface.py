import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import math

class SphereFace(nn.Module):
    """
    Implementation of sphereface
    Implementation taken from: https://github.com/4uiiurz1/pytorch-adacos 
    """

    def __init__(self, in_features, out_features, ls_eps=0, s=30.0, m=1.35):
        super(SphereFace, self).__init__()
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
        
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(self.m * theta)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features

        output = logits * (1 - one_hot) + target_logits * one_hot
        output *= self.s

        return output