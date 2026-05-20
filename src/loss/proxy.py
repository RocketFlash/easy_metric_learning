import torch
import torch.nn as nn
import torch.nn.functional as F


def _labels_from_targets(targets):
    if isinstance(targets, (list, tuple)):
        return targets[0]
    return targets


def _masked_logsumexp_with_zero(logits, mask, dim):
    masked_logits = logits.masked_fill(~mask, -torch.inf)
    logsumexp = torch.logsumexp(masked_logits, dim=dim)
    return torch.logaddexp(torch.zeros_like(logsumexp), logsumexp)


class ProxyAnchorLoss(nn.Module):
    def __init__(self, in_features, out_features, margin=0.1, alpha=32):
        super(ProxyAnchorLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.margin = margin
        self.alpha = alpha
        self.proxies = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_normal_(self.proxies, mode="fan_out")

    def forward(self, embeddings, labels):
        labels = _labels_from_targets(labels).view(-1)
        embeddings = F.normalize(embeddings)
        proxies = F.normalize(self.proxies)
        cosine = F.linear(embeddings, proxies).t()

        one_hot = F.one_hot(labels, num_classes=self.out_features).t().float()
        positive_mask = one_hot > 0
        negative_mask = ~positive_mask
        valid_positive_proxy = positive_mask.any(dim=1)

        if not valid_positive_proxy.any():
            return embeddings.sum() * 0 + self.proxies.sum() * 0

        positive_logits = -self.alpha * (cosine - self.margin)
        negative_logits = self.alpha * (cosine + self.margin)

        positive_loss = _masked_logsumexp_with_zero(
            positive_logits,
            positive_mask,
            dim=1,
        )
        positive_loss = positive_loss[valid_positive_proxy].mean()
        negative_loss = _masked_logsumexp_with_zero(
            negative_logits,
            negative_mask,
            dim=1,
        ).mean()
        return positive_loss + negative_loss


class ProxyNCALoss(nn.Module):
    def __init__(self, in_features, out_features, temperature=0.1):
        super(ProxyNCALoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.temperature = temperature
        self.proxies = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_normal_(self.proxies, mode="fan_out")

    def forward(self, embeddings, labels):
        labels = _labels_from_targets(labels).view(-1)
        embeddings = F.normalize(embeddings)
        proxies = F.normalize(self.proxies)
        distances = torch.cdist(embeddings, proxies, p=2).pow(2)
        logits = -distances / self.temperature
        return F.cross_entropy(logits, labels)
