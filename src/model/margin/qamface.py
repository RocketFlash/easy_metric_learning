import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from .utils import build_one_hot


class QAMFace(nn.Module):
    """
    Quality-adaptive angular margin head.

    This complements MagFace/AdaFace by mapping feature magnitude to a bounded
    quality score, then using that score to interpolate the target angular
    margin and scale for each sample.
    """

    def __init__(
        self,
        in_features,
        out_features,
        s=64.0,
        min_s=16.0,
        l_a=10.0,
        u_a=110.0,
        l_margin=0.2,
        u_margin=0.6,
        easy_margin=False,
        ls_eps=0.0,
    ):
        super(QAMFace, self).__init__()
        if u_a <= l_a:
            raise ValueError("u_a must be greater than l_a")
        if u_margin < l_margin:
            raise ValueError("u_margin must be greater than or equal to l_margin")
        if s < min_s:
            raise ValueError("s must be greater than or equal to min_s")

        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.min_s = min_s
        self.l_a = l_a
        self.u_a = u_a
        self.l_margin = l_margin
        self.u_margin = u_margin
        self.easy_margin = easy_margin
        self.ls_eps = ls_eps
        self.m = (l_margin, u_margin)
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def _quality(self, x):
        x_norm = torch.norm(x, dim=1, keepdim=True).clamp(self.l_a, self.u_a)
        return ((x_norm - self.l_a) / (self.u_a - self.l_a)).clamp(0.0, 1.0)

    def forward(self, x, label):
        quality = self._quality(x)
        margin = self.l_margin + (self.u_margin - self.l_margin) * quality
        scale = self.min_s + (self.s - self.min_s) * quality

        cosine = F.linear(F.normalize(x), F.normalize(self.weight)).clamp(-1.0, 1.0)
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
        phi = cosine * torch.cos(margin) - sine * torch.sin(margin)

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            threshold = torch.cos(math.pi - margin)
            mm = torch.sin(math.pi - margin) * margin
            phi = torch.where(cosine > threshold, phi, cosine - mm)

        one_hot = build_one_hot(
            label, self.out_features, device=cosine.device, dtype=cosine.dtype
        )
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features

        return scale * (one_hot * phi + (1.0 - one_hot) * cosine)
