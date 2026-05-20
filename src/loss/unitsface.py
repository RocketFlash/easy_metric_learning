import torch
import torch.nn as nn
import torch.nn.functional as F


def _labels_from_targets(targets):
    if isinstance(targets, (list, tuple)):
        return targets[0]
    return targets


class UniTSFaceLoss(nn.Module):
    """
    Unified threshold sample-to-sample loss.

    This loss complements sample-to-class margin heads by applying separate
    thresholds to positive and negative sample pairs inside the batch.
    """

    def __init__(
        self,
        positive_threshold=0.5,
        negative_threshold=0.2,
        scale=32.0,
    ):
        super(UniTSFaceLoss, self).__init__()
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.scale = scale

    def forward(self, embeddings, labels):
        labels = _labels_from_targets(labels).view(-1)
        embeddings = F.normalize(embeddings)
        similarity = torch.matmul(embeddings, embeddings.t())
        same = labels[:, None].eq(labels[None, :])
        eye = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
        positive_mask = same & ~eye
        negative_mask = ~same

        if positive_mask.sum() == 0 or negative_mask.sum() == 0:
            return embeddings.sum() * 0

        positive_logits = self.scale * (
            self.positive_threshold - similarity[positive_mask]
        )
        negative_logits = self.scale * (
            similarity[negative_mask] - self.negative_threshold
        )
        positive_loss = F.softplus(positive_logits).mean()
        negative_loss = F.softplus(negative_logits).mean()
        return positive_loss + negative_loss
