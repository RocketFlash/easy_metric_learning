import torch
import torch.nn as nn
import torch.nn.functional as F


def _logsumexp_with_zero(logits):
    return torch.logsumexp(torch.cat([logits.new_zeros(1), logits]), dim=0)


class MultiSimilarityLoss(nn.Module):
    """
    Multi-Similarity loss over normalized batch embeddings.

    Use with loss config field `input: embeddings` and a PK-style sampler so
    each batch contains both positive and negative pairs.
    """

    def __init__(self, alpha=2.0, beta=50.0, base=0.5, eps=1e-5):
        super(MultiSimilarityLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.base = base
        self.eps = eps

    def forward(self, embeddings, labels):
        if isinstance(labels, (list, tuple)):
            labels = labels[0]

        embeddings = F.normalize(embeddings)
        similarity = torch.matmul(embeddings, embeddings.t())
        labels = labels.view(-1)
        losses = []

        for anchor_idx in range(embeddings.size(0)):
            positive_mask = labels.eq(labels[anchor_idx])
            positive_mask[anchor_idx] = False
            negative_mask = labels.ne(labels[anchor_idx])

            positive_similarity = similarity[anchor_idx][positive_mask]
            negative_similarity = similarity[anchor_idx][negative_mask]

            if positive_similarity.numel() == 0 or negative_similarity.numel() == 0:
                continue

            hard_positive = positive_similarity[
                positive_similarity < negative_similarity.max() + self.eps
            ]
            hard_negative = negative_similarity[
                negative_similarity > positive_similarity.min() - self.eps
            ]

            if hard_positive.numel() == 0 or hard_negative.numel() == 0:
                continue

            positive_logits = -self.alpha * (hard_positive - self.base)
            negative_logits = self.beta * (hard_negative - self.base)
            positive_loss = _logsumexp_with_zero(positive_logits) / self.alpha
            negative_loss = _logsumexp_with_zero(negative_logits) / self.beta
            losses.append(positive_loss + negative_loss)

        if not losses:
            return embeddings.sum() * 0

        return torch.stack(losses).mean()
