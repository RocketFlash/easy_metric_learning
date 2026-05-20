import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class UnifiedCrossEntropyLoss(nn.Module):
    """UniFace UCE loss with optional sampled negative classes."""

    def __init__(
        self,
        in_features,
        out_features,
        m=0.4,
        s=64.0,
        l=1.0,
        r=None,
        sample_rate=1.0,
        min_sample_classes=0,
        bias_scale=10.0,
        logit_clip=50.0,
    ):
        super(UnifiedCrossEntropyLoss, self).__init__()
        if out_features <= 0:
            raise ValueError("out_features must be positive")
        if in_features <= 0:
            raise ValueError("in_features must be positive")

        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.l = l
        self.sample_rate = sample_rate if r is None else r
        self.min_sample_classes = min_sample_classes
        self.logit_clip = logit_clip

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(
            torch.tensor(
                [math.log(out_features * self.sample_rate * bias_scale)],
                dtype=torch.float32,
            )
        )
        self.register_buffer("last_sampled_indices", torch.empty(0, dtype=torch.long))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def _sample_class_indices(self, labels):
        positive = torch.unique(labels, sorted=True)
        num_sample = int(math.ceil(self.out_features * self.sample_rate))
        num_sample = max(num_sample, self.min_sample_classes, positive.numel())
        num_sample = min(num_sample, self.out_features)

        if not self.training or num_sample >= self.out_features:
            return torch.arange(self.out_features, device=labels.device)

        negative_mask = torch.ones(
            self.out_features,
            dtype=torch.bool,
            device=labels.device,
        )
        negative_mask[positive] = False
        negative = torch.nonzero(negative_mask, as_tuple=False).flatten()
        n_negative = num_sample - positive.numel()

        if n_negative > 0:
            perm = torch.randperm(negative.numel(), device=labels.device)[:n_negative]
            sampled = torch.cat([positive, negative[perm]])
        else:
            sampled = positive

        return sampled.sort()[0]

    def forward(self, embeddings, labels):
        labels = labels.reshape(-1).long()
        valid_mask = labels != -1
        if valid_mask.sum() == 0:
            return embeddings.sum() * 0.0

        embeddings = embeddings[valid_mask]
        labels = labels[valid_mask]
        if labels.min() < 0 or labels.max() >= self.out_features:
            raise ValueError("labels must be in [0, out_features)")

        sampled = self._sample_class_indices(labels)
        self.last_sampled_indices = sampled.detach()

        norm_embeddings = F.normalize(embeddings)
        norm_weight = F.normalize(self.weight[sampled])
        cosine = F.linear(norm_embeddings, norm_weight).clamp(-1.0, 1.0)

        mapped_labels = torch.searchsorted(sampled, labels)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, mapped_labels.view(-1, 1), 1.0)

        positive_logits = (self.s * (cosine - self.m) - self.bias).clamp(
            -self.logit_clip,
            self.logit_clip,
        )
        negative_logits = (self.s * cosine - self.bias).clamp(
            -self.logit_clip,
            self.logit_clip,
        )

        positive_loss = F.softplus(-positive_logits) * one_hot
        negative_loss = F.softplus(negative_logits) * (1.0 - one_hot) * self.l
        return (positive_loss + negative_loss).sum(dim=1).mean()
