import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _labels_from_targets(targets):
    if isinstance(targets, (list, tuple)):
        return targets[0]
    return targets


class BoundaryFaceLoss(nn.Module):
    """
    BoundaryFace-style class-boundary mining with optional closed-set correction.
    """

    def __init__(
        self,
        in_features,
        out_features,
        s=64.0,
        m=0.5,
        boundary_margin=0.05,
        correction_threshold=0.35,
        correction_weight=0.2,
        easy_margin=False,
    ):
        super(BoundaryFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.boundary_margin = boundary_margin
        self.correction_threshold = correction_threshold
        self.correction_weight = correction_weight
        self.easy_margin = easy_margin
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def _arc_logits(self, embeddings, labels):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight)).clamp(
            -1.0, 1.0
        )
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = F.one_hot(labels, num_classes=self.out_features).float()
        return self.s * (one_hot * phi + (1.0 - one_hot) * cosine), cosine

    def forward(self, embeddings, labels):
        labels = _labels_from_targets(labels).view(-1).long()
        logits, cosine = self._arc_logits(embeddings, labels)
        ce_loss = F.cross_entropy(logits, labels)

        one_hot = F.one_hot(labels, num_classes=self.out_features).bool()
        target_cosine = cosine[one_hot]
        negative_cosine = cosine.masked_fill(one_hot, -2.0)
        nearest_negative_cosine, nearest_negative_label = negative_cosine.max(dim=1)

        boundary_violation = F.relu(
            nearest_negative_cosine - target_cosine + self.boundary_margin
        )
        boundary_loss = boundary_violation.mean()

        correction_mask = (
            nearest_negative_cosine - target_cosine > self.correction_threshold
        )
        if correction_mask.any():
            corrected_logits, _ = self._arc_logits(
                embeddings[correction_mask],
                nearest_negative_label[correction_mask],
            )
            correction_loss = F.cross_entropy(
                corrected_logits,
                nearest_negative_label[correction_mask],
            )
        else:
            correction_loss = embeddings.sum() * 0

        return ce_loss + boundary_loss + self.correction_weight * correction_loss
