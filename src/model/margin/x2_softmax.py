import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from .utils import build_one_hot, get_primary_label


class X2Softmax(nn.Module):
    """
    Adaptive angular-margin softmax inspired by X2-Softmax.

    The target margin is derived from the angular separation between the target
    class center and the remaining class centers, so easier, better-separated
    classes receive a larger angular penalty while crowded classes receive a
    smaller one.
    """

    def __init__(
        self,
        in_features,
        out_features,
        s=64.0,
        m=0.5,
        min_m=0.0,
        easy_margin=False,
        ls_eps=0.0,
    ):
        super(X2Softmax, self).__init__()
        if isinstance(m, dict):
            raise ValueError("X2Softmax does not support dynamic margin dictionaries")

        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.min_m = min_m
        self.easy_margin = easy_margin
        self.ls_eps = ls_eps
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def _adaptive_margin(self, labels):
        normalized_weight = F.normalize(self.weight)
        class_cosine = torch.matmul(normalized_weight, normalized_weight.t()).clamp(
            -1.0, 1.0
        )
        class_angles = torch.acos(class_cosine)
        eye = torch.eye(self.out_features, dtype=torch.bool, device=class_angles.device)
        class_angles = class_angles.masked_fill(eye, float("nan"))
        target_angle = torch.nanmean(class_angles[labels.long()], dim=1)
        normalized_angle = (target_angle / math.pi).clamp(0.0, 1.0)
        return self.min_m + (self.m - self.min_m) * normalized_angle.pow(2)

    def forward(self, x, label):
        primary_label = get_primary_label(label)
        adaptive_margin = self._adaptive_margin(primary_label).view(-1, 1)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight)).clamp(-1.0, 1.0)
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))

        phi = cosine * torch.cos(adaptive_margin) - sine * torch.sin(adaptive_margin)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            threshold = torch.cos(math.pi - adaptive_margin)
            mm = torch.sin(math.pi - adaptive_margin) * adaptive_margin
            phi = torch.where(cosine > threshold, phi, cosine - mm)

        one_hot = build_one_hot(
            label, self.out_features, device=cosine.device, dtype=cosine.dtype
        )
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features

        return self.s * (one_hot * phi + (1.0 - one_hot) * cosine)

    def update(self, m=0.5):
        if isinstance(m, dict):
            raise ValueError("X2Softmax does not support dynamic margin dictionaries")
        self.m = m
