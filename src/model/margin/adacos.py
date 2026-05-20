import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import math
from .utils import build_one_hot


class AdaCos(nn.Module):
    """
    Implementation of adacos
    Implementation taken from: https://github.com/4uiiurz1/pytorch-adacos
    """

    def __init__(
        self, in_features, out_features, m=0.50, ls_eps=0, theta_zero=math.pi / 4
    ):
        super(AdaCos, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.theta_zero = theta_zero
        self.register_buffer(
            "s",
            torch.tensor(
                math.log(out_features - 1) / math.cos(theta_zero),
                dtype=torch.float32,
            ),
        )
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input_x, label):
        x = F.normalize(input_x)
        W = F.normalize(self.weight)
        logits = F.linear(x, W)

        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(theta + self.m)
        one_hot = build_one_hot(
            label, self.out_features, device=logits.device, dtype=logits.dtype
        )
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        output = logits * (1 - one_hot) + target_logits * one_hot

        with torch.no_grad():
            B_avg = torch.where(
                one_hot < 1,
                torch.exp(self.s * logits.float()),
                torch.zeros_like(logits.float()),
            )
            B_avg = torch.sum(B_avg) / input_x.size(0)
            theta_med = torch.median(theta)
            updated_s = torch.log(B_avg) / torch.cos(
                torch.min(self.theta_zero * torch.ones_like(theta_med), theta_med)
            )
            self.s.copy_(updated_s.to(dtype=self.s.dtype))
        output *= self.s

        return output

    def update(self, m=0.5):
        self.m = m
