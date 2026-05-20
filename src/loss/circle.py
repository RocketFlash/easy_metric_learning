import torch
import torch.nn as nn
import torch.nn.functional as F


class CircleLoss(nn.Module):
    """
    Pair-level Circle Loss over embeddings in a batch.

    Use with loss config field `input: embeddings` so the trainer passes
    embeddings instead of classification logits.
    """

    def __init__(self, m=0.25, gamma=256):
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, embeddings, labels):
        if isinstance(labels, (list, tuple)):
            labels = labels[0]

        embeddings = F.normalize(embeddings)
        similarity = torch.matmul(embeddings, embeddings.t())
        labels = labels.view(-1, 1)

        positive_mask = labels.eq(labels.t()).triu(diagonal=1)
        negative_mask = labels.ne(labels.t()).triu(diagonal=1)
        positive_similarity = similarity[positive_mask]
        negative_similarity = similarity[negative_mask]

        if positive_similarity.numel() == 0 or negative_similarity.numel() == 0:
            return embeddings.sum() * 0

        alpha_p = torch.clamp_min(-positive_similarity.detach() + 1 + self.m, min=0.0)
        alpha_n = torch.clamp_min(negative_similarity.detach() + self.m, min=0.0)
        delta_p = 1 - self.m
        delta_n = self.m

        logits_p = -self.gamma * alpha_p * (positive_similarity - delta_p)
        logits_n = self.gamma * alpha_n * (negative_similarity - delta_n)

        return self.soft_plus(
            torch.logsumexp(logits_p, dim=0) + torch.logsumexp(logits_n, dim=0)
        )
