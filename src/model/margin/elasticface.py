import torch
from torch import nn

from .utils import build_one_hot, get_primary_label, l2_norm


def _build_elastic_margin_hot(label, cos_theta, margin_value, std, plus):
    primary_label = get_primary_label(label).view(-1).long()
    valid_index = torch.where(primary_label != -1)[0]
    m_hot = torch.zeros_like(cos_theta)
    if valid_index.numel() == 0:
        return m_hot

    margin = torch.normal(
        mean=margin_value,
        std=std,
        size=(valid_index.numel(), 1),
        device=cos_theta.device,
    ).to(dtype=cos_theta.dtype)
    valid_labels = primary_label[valid_index]
    if plus:
        with torch.no_grad():
            target_cosine = cos_theta[valid_index, valid_labels].detach().clone()
            _, cosine_order = torch.sort(target_cosine, dim=0, descending=True)
            sorted_margin, _ = torch.sort(margin.view(-1), dim=0, descending=False)
            assigned_margin = torch.empty_like(sorted_margin)
            assigned_margin[cosine_order] = sorted_margin
        margin = assigned_margin.view(-1, 1)

    if isinstance(label, (list, tuple)):
        one_hot = build_one_hot(
            label, cos_theta.size(1), device=cos_theta.device, dtype=cos_theta.dtype
        )
    else:
        one_hot = torch.zeros_like(cos_theta)
        one_hot[valid_index, valid_labels] = 1
    m_hot[valid_index] = one_hot[valid_index] * margin
    return m_hot


class ElasticArcFace(nn.Module):
    """
    Code from https://github.com/fdbtrs/ElasticFace
    """

    def __init__(
        self, in_features, out_features, s=64.0, m=0.50, std=0.0125, plus=False
    ):
        super(ElasticArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.kernel = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.normal_(self.kernel, std=0.01)
        self.std = std
        self.plus = plus

    def forward(self, embbedings, label):
        embbedings = l2_norm(embbedings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        m_hot = _build_elastic_margin_hot(label, cos_theta, self.m, self.std, self.plus)
        cos_theta.acos_()
        cos_theta += m_hot
        cos_theta.cos_().mul_(self.s)
        return cos_theta


class ElasticCosFace(nn.Module):
    """
    Code from https://github.com/fdbtrs/ElasticFace
    """

    def __init__(
        self, in_features, out_features, s=64.0, m=0.35, std=0.0125, plus=False
    ):
        super(ElasticCosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.kernel = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.normal_(self.kernel, std=0.01)
        self.std = std
        self.plus = plus

    def forward(self, embbedings, label):
        embbedings = l2_norm(embbedings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        m_hot = _build_elastic_margin_hot(label, cos_theta, self.m, self.std, self.plus)
        cos_theta -= m_hot
        ret = cos_theta * self.s
        return ret
