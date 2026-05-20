import torch
import torch.nn as nn
import torch.nn.functional as F


def _labels_from_targets(targets):
    if isinstance(targets, (list, tuple)):
        return targets[0]
    return targets


class SFaceLoss(nn.Module):
    """
    Sigmoid-constrained hypersphere loss with trainable class centers.
    """

    def __init__(
        self,
        in_features,
        out_features,
        s=64.0,
        margin=0.4,
        gamma=20.0,
    ):
        super(SFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.margin = margin
        self.gamma = gamma
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        labels = _labels_from_targets(labels).view(-1)
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight)).clamp(
            -1.0, 1.0
        )
        one_hot = F.one_hot(labels, num_classes=self.out_features).bool()
        positive = cosine[one_hot]
        negative = cosine[~one_hot].view(cosine.size(0), -1)

        positive_loss = F.softplus(-self.gamma * (positive - self.margin))
        negative_loss = F.softplus(self.gamma * (negative + self.margin)).mean(dim=1)
        return (positive_loss + negative_loss).mean()
