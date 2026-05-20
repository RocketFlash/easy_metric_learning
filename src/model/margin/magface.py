import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from .utils import build_one_hot


class MagFace(nn.Module):
    """
    Magnitude-aware angular margin from MagFace.

    The forward pass returns standard classification logits for the existing
    cross-entropy training path. The MagFace magnitude regularizer is exposed
    through regularization_loss() and is added by the trainers when present.
    """

    regularization_loss_name = "magface_reg"

    def __init__(
        self,
        in_features,
        out_features,
        s=64.0,
        l_a=10.0,
        u_a=110.0,
        l_margin=0.45,
        u_margin=0.8,
        lambda_g=35.0,
        easy_margin=True,
        ls_eps=0.0,
    ):
        super(MagFace, self).__init__()
        if u_a <= l_a:
            raise ValueError("u_a must be greater than l_a")
        if u_margin < l_margin:
            raise ValueError("u_margin must be greater than or equal to l_margin")

        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.l_a = l_a
        self.u_a = u_a
        self.l_margin = l_margin
        self.u_margin = u_margin
        self.lambda_g = lambda_g
        self.easy_margin = easy_margin
        self.ls_eps = ls_eps
        self.m = (l_margin, u_margin)

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self._regularization_loss = None

    def _adaptive_margin(self, x_norm):
        margin_slope = (self.u_margin - self.l_margin) / (self.u_a - self.l_a)
        return margin_slope * (x_norm - self.l_a) + self.l_margin

    def _calc_regularization_loss(self, x_norm):
        g = x_norm / (self.u_a**2) + 1 / x_norm
        return self.lambda_g * torch.mean(g)

    def forward(self, x, label):
        x_norm = torch.norm(x, dim=1, keepdim=True).clamp(self.l_a, self.u_a)
        ada_margin = self._adaptive_margin(x_norm)

        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        cosine = cosine.clamp(-1, 1)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))

        phi = cosine * torch.cos(ada_margin) - sine * torch.sin(ada_margin)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            threshold = torch.cos(math.pi - ada_margin)
            mm = torch.sin(math.pi - ada_margin) * ada_margin
            phi = torch.where(cosine > threshold, phi, cosine - mm)

        one_hot = build_one_hot(
            label, self.out_features, device=cosine.device, dtype=cosine.dtype
        )
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        self._regularization_loss = self._calc_regularization_loss(x_norm)

        return output

    def regularization_loss(self):
        return self._regularization_loss
