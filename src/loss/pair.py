import torch
import torch.nn as nn
import torch.nn.functional as F


def _labels_from_targets(targets):
    if isinstance(targets, (list, tuple)):
        return targets[0]
    return targets


def pairwise_cosine(embeddings):
    embeddings = F.normalize(embeddings)
    return torch.matmul(embeddings, embeddings.t())


def pairwise_distance(embeddings, squared=False, eps=1e-12):
    distances = torch.cdist(embeddings, embeddings, p=2)
    if squared:
        distances = distances.pow(2)
    return distances.clamp_min(eps)


class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin=0.2, squared=False):
        super(BatchHardTripletLoss, self).__init__()
        self.margin = margin
        self.squared = squared

    def forward(self, embeddings, labels):
        labels = _labels_from_targets(labels).view(-1)
        distances = pairwise_distance(embeddings, squared=self.squared)
        same = labels[:, None].eq(labels[None, :])
        eye = torch.eye(labels.numel(), dtype=torch.bool, device=labels.device)
        positive_mask = same & ~eye
        negative_mask = ~same

        if positive_mask.sum() == 0 or negative_mask.sum() == 0:
            return embeddings.sum() * 0

        hardest_positive = distances.masked_fill(~positive_mask, 0.0).max(dim=1).values
        hardest_negative = (
            distances.masked_fill(~negative_mask, float("inf")).min(dim=1).values
        )
        valid_anchor = positive_mask.any(dim=1) & negative_mask.any(dim=1)
        losses = F.relu(
            hardest_positive[valid_anchor]
            - hardest_negative[valid_anchor]
            + self.margin
        )
        if losses.numel() == 0:
            return embeddings.sum() * 0
        return losses.mean()


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, embeddings, labels):
        labels = _labels_from_targets(labels).view(-1)
        embeddings = F.normalize(embeddings)
        logits = torch.matmul(embeddings, embeddings.t()) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        mask = labels[:, None].eq(labels[None, :]).float()
        logits_mask = torch.ones_like(mask) - torch.eye(
            mask.size(0), device=mask.device
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
        positive_count = mask.sum(dim=1)
        valid = positive_count > 0
        if not valid.any():
            return embeddings.sum() * 0

        mean_log_prob_pos = (mask * log_prob).sum(dim=1)[valid] / positive_count[valid]
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.mean()


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        labels = _labels_from_targets(labels).view(-1)
        return SupervisedContrastiveLoss(
            temperature=self.temperature,
            base_temperature=self.temperature,
        )(embeddings, labels)
