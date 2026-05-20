import torch
import torch.nn as nn
from torch.nn import Parameter
import math
from .utils import build_one_hot, get_primary_label, l2_norm


class CurricularFace(nn.Module):
    """
    Based on code from https://github.com/HuangYG123/CurricularFace
    """

    def __init__(self, in_features, out_features, ls_eps=0, m=0.5, s=64.0):
        super(CurricularFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.ls_eps = ls_eps
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.kernel = Parameter(torch.Tensor(in_features, out_features))
        self.register_buffer("t", torch.zeros(1))
        nn.init.normal_(self.kernel, std=0.01)

    def forward(self, embbedings, label):
        primary_label = get_primary_label(label).view(-1).long()
        embbedings = l2_norm(embbedings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        target_logit = cos_theta[
            torch.arange(0, embbedings.size(0), device=cos_theta.device),
            primary_label,
        ].view(-1, 1)

        sin_theta = torch.sqrt((1.0 - torch.pow(cos_theta, 2)).clamp(0, 1))
        cos_theta_m = (
            cos_theta * self.cos_m - sin_theta * self.sin_m
        )  # cos(target+margin)
        primary_cos_theta_m = cos_theta_m[
            torch.arange(0, embbedings.size(0), device=cos_theta.device),
            primary_label,
        ].view(-1, 1)
        mask = cos_theta > primary_cos_theta_m
        final_target_logit = torch.where(
            cos_theta > self.threshold, cos_theta_m, cos_theta - self.mm
        )

        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        cos_theta[mask] = hard_example * (self.t + hard_example)

        one_hot = build_one_hot(
            label, self.out_features, device=cos_theta.device, dtype=cos_theta.dtype
        )

        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features

        cos_theta = one_hot * final_target_logit + (1.0 - one_hot) * cos_theta

        output = cos_theta * self.s
        return output
