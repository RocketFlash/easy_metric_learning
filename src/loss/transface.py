import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransFaceEntropyMiningLoss(nn.Module):
    """Entropy-weighted cross entropy inspired by TransFace EHSM."""

    def __init__(
        self,
        gamma=1.0,
        hard_fraction=1.0,
        easy_fraction=0.0,
        easy_weight=1.0,
        weight_by_entropy=True,
        ignore_index=-100,
        eps=1e-12,
    ):
        super(TransFaceEntropyMiningLoss, self).__init__()
        if not 0 < hard_fraction <= 1:
            raise ValueError("hard_fraction must be in (0, 1]")
        if not 0 <= easy_fraction <= 1:
            raise ValueError("easy_fraction must be in [0, 1]")

        self.gamma = gamma
        self.hard_fraction = hard_fraction
        self.easy_fraction = easy_fraction
        self.easy_weight = easy_weight
        self.weight_by_entropy = weight_by_entropy
        self.ignore_index = ignore_index
        self.eps = eps

    def _select_samples(self, losses):
        n_samples = losses.numel()
        if self.hard_fraction == 1.0 and self.easy_fraction == 0.0:
            return losses

        selected = []
        hard_k = max(1, int(math.ceil(n_samples * self.hard_fraction)))
        selected.append(torch.topk(losses, hard_k, largest=True).values)

        if self.easy_fraction > 0:
            easy_k = max(1, int(math.ceil(n_samples * self.easy_fraction)))
            selected.append(
                torch.topk(losses, easy_k, largest=False).values * self.easy_weight
            )

        return torch.cat(selected)

    def forward(self, logits, labels):
        labels = labels.reshape(-1).long()
        valid_mask = (labels != self.ignore_index) & (labels != -1)
        if valid_mask.sum() == 0:
            return logits.sum() * 0.0

        logits = logits[valid_mask]
        labels = labels[valid_mask]
        per_sample_loss = F.cross_entropy(logits, labels, reduction="none")

        if self.weight_by_entropy:
            probabilities = F.softmax(logits.detach(), dim=1)
            entropy = -(
                probabilities * torch.log(probabilities.clamp_min(self.eps))
            ).sum(dim=1)
            per_sample_loss = per_sample_loss * (1.0 + torch.exp(-self.gamma * entropy))

        return self._select_samples(per_sample_loss).mean()
