import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from .utils import build_one_hot


class CircleMargin(nn.Module):
    """
    Class-level Circle Loss margin head.

    This produces logits that can be consumed by the existing cross-entropy
    losses, following the class-level Circle Loss formulation.
    """

    def __init__(self, in_features, out_features, s=256.0, m=0.25, ls_eps=0.0):
        super(CircleMargin, self).__init__()
        if isinstance(m, dict):
            raise ValueError(
                "CircleMargin does not support dynamic margin dictionaries"
            )

        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, label):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))

        alpha_p = torch.clamp_min(-cosine.detach() + 1 + self.m, min=0.0)
        alpha_n = torch.clamp_min(cosine.detach() + self.m, min=0.0)
        delta_p = 1 - self.m
        delta_n = self.m

        positive_logits = self.s * alpha_p * (cosine - delta_p)
        negative_logits = self.s * alpha_n * (cosine - delta_n)

        one_hot = build_one_hot(
            label, self.out_features, device=cosine.device, dtype=cosine.dtype
        )
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features

        return one_hot * positive_logits + (1.0 - one_hot) * negative_logits

    def update(self, m=0.25):
        if isinstance(m, dict):
            raise ValueError(
                "CircleMargin does not support dynamic margin dictionaries"
            )
        self.m = m
