import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from .utils import build_one_hot, get_primary_label, is_mixed_label


class PartialFCArcMarginProduct(nn.Module):
    """
    Sampled-class ArcFace head inspired by PartialFC.

    This keeps the repo's existing CrossEntropy target contract by scattering
    sampled logits back into full class space. It reduces sampled-class matmul
    compute, but does not implement distributed classifier sharding.
    """

    def __init__(
        self,
        in_features,
        out_features,
        s=64.0,
        m=0.5,
        sample_rate=0.1,
        min_sample_classes=1024,
        easy_margin=False,
        ls_eps=0.0,
        unsampled_logit=-10000.0,
    ):
        super(PartialFCArcMarginProduct, self).__init__()
        if isinstance(m, dict):
            raise ValueError(
                "PartialFCArcMarginProduct does not support dynamic margins"
            )
        if not 0 < sample_rate <= 1:
            raise ValueError("sample_rate must be in the interval (0, 1]")

        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.sample_rate = sample_rate
        self.min_sample_classes = min_sample_classes
        self.easy_margin = easy_margin
        self.ls_eps = ls_eps
        self.unsampled_logit = unsampled_logit

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self._set_margin_constants(m)

    def _set_margin_constants(self, m):
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def _positive_labels(self, label):
        if is_mixed_label(label):
            label1, label2, _ = label
            return torch.unique(torch.cat([label1.view(-1), label2.view(-1)]).long())
        return torch.unique(label.view(-1).long())

    def _sample_classes(self, label, device):
        positive_labels = self._positive_labels(label).to(device)
        sample_count = max(
            int(self.out_features * self.sample_rate),
            int(positive_labels.numel()),
            min(self.min_sample_classes, self.out_features),
        )
        sample_count = min(sample_count, self.out_features)

        if sample_count == self.out_features:
            return torch.arange(self.out_features, device=device, dtype=torch.long)

        negative_mask = torch.ones(self.out_features, device=device, dtype=torch.bool)
        negative_mask[positive_labels] = False
        negative_labels = negative_mask.nonzero(as_tuple=False).view(-1)
        negative_count = sample_count - positive_labels.numel()
        negative_indices = torch.randperm(negative_labels.numel(), device=device)[
            :negative_count
        ]
        sampled = torch.cat([positive_labels, negative_labels[negative_indices]])
        return torch.sort(sampled).values

    def _remap_labels(self, label, sampled_classes):
        if is_mixed_label(label):
            label1, label2, lam = label
            return [
                torch.searchsorted(sampled_classes, label1.long()),
                torch.searchsorted(sampled_classes, label2.long()),
                lam,
            ]
        primary_label = get_primary_label(label)
        return torch.searchsorted(sampled_classes, primary_label.long())

    def forward(self, x, label):
        sampled_classes = self._sample_classes(label, x.device)
        sampled_weight = self.weight.index_select(0, sampled_classes)

        cosine = F.linear(F.normalize(x), F.normalize(sampled_weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        remapped_label = self._remap_labels(label, sampled_classes)
        one_hot = build_one_hot(
            remapped_label,
            sampled_classes.numel(),
            device=cosine.device,
            dtype=cosine.dtype,
        )
        if self.ls_eps > 0:
            one_hot = (
                1 - self.ls_eps
            ) * one_hot + self.ls_eps / sampled_classes.numel()

        sampled_output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        sampled_output *= self.s

        output = cosine.new_full(
            (x.size(0), self.out_features),
            self.unsampled_logit,
        )
        output[:, sampled_classes] = sampled_output
        return output

    def update(self, m=0.5):
        if isinstance(m, dict):
            raise ValueError(
                "PartialFCArcMarginProduct does not support dynamic margins"
            )
        self.m = m
        self._set_margin_constants(m)


def _distributed_info():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _local_class_count(out_features, rank, world_size):
    return out_features // world_size + int(rank < out_features % world_size)


def _local_class_start(out_features, rank, world_size):
    return out_features // world_size * rank + min(rank, out_features % world_size)


def _all_gather_logits(local_logits, out_features, unsampled_logit):
    rank, world_size = _distributed_info()
    if world_size == 1:
        return local_logits

    from torch.distributed.nn.functional import all_gather

    max_local = max(
        _local_class_count(out_features, rank_i, world_size)
        for rank_i in range(world_size)
    )
    if local_logits.size(1) < max_local:
        local_logits = F.pad(
            local_logits,
            (0, max_local - local_logits.size(1)),
            value=unsampled_logit,
        )

    gathered = all_gather(local_logits)
    logits = []
    for rank_i, rank_logits in enumerate(gathered):
        local_count = _local_class_count(out_features, rank_i, world_size)
        logits.append(rank_logits[:, :local_count])
    return torch.cat(logits, dim=1)


def _all_gather_equal_batch(tensor):
    rank, world_size = _distributed_info()
    if world_size == 1:
        return tensor, 0, tensor.size(0)

    from torch.distributed.nn.functional import all_gather

    local_batch_size = tensor.size(0)
    gathered = all_gather(tensor.contiguous())
    global_tensor = torch.cat(gathered, dim=0)
    return global_tensor, rank * local_batch_size, local_batch_size


def _all_gather_equal_labels(labels):
    rank, world_size = _distributed_info()
    labels = labels.reshape(-1).long()
    if world_size == 1:
        return labels, 0, labels.size(0)

    local_batch_size = labels.size(0)
    gathered = [torch.empty_like(labels) for _ in range(world_size)]
    dist.all_gather(gathered, labels.contiguous())
    global_labels = torch.cat(gathered, dim=0)
    return global_labels, rank * local_batch_size, local_batch_size


class DistributedPartialFCArcMarginProduct(nn.Module):
    """Distributed sharded PartialFC ArcFace head.

    Each rank stores only its local classifier shard. Local sampled logits are
    gathered back into global class order so existing CrossEntropy targets keep
    working.
    """

    def __init__(
        self,
        in_features,
        out_features,
        s=64.0,
        m=0.5,
        sample_rate=1.0,
        min_sample_classes=1024,
        easy_margin=False,
        ls_eps=0.0,
        unsampled_logit=-10000.0,
    ):
        super(DistributedPartialFCArcMarginProduct, self).__init__()
        if isinstance(m, dict):
            raise ValueError(
                "DistributedPartialFCArcMarginProduct does not support dynamic margins"
            )
        if not 0 < sample_rate <= 1:
            raise ValueError("sample_rate must be in the interval (0, 1]")

        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.sample_rate = sample_rate
        self.min_sample_classes = min_sample_classes
        self.easy_margin = easy_margin
        self.ls_eps = ls_eps
        self.unsampled_logit = unsampled_logit

        self.rank, self.world_size = _distributed_info()
        self.class_start = _local_class_start(out_features, self.rank, self.world_size)
        self.local_out_features = _local_class_count(
            out_features,
            self.rank,
            self.world_size,
        )
        self.class_end = self.class_start + self.local_out_features

        self.weight = Parameter(torch.FloatTensor(self.local_out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self._set_margin_constants(m)

    def _sync_distributed_info(self):
        rank, world_size = _distributed_info()
        if rank == self.rank and world_size == self.world_size:
            return
        if self.weight.size(0) != self.out_features:
            raise RuntimeError(
                "DistributedPartialFCArcMarginProduct cannot change distributed "
                "world size after constructing sharded classifier weights."
            )
        self.rank = rank
        self.world_size = world_size
        self.class_start = _local_class_start(
            self.out_features, self.rank, self.world_size
        )
        self.local_out_features = _local_class_count(
            self.out_features,
            self.rank,
            self.world_size,
        )
        self.class_end = self.class_start + self.local_out_features

    def _select_sampled_weight(self, sampled_global_classes, sampled_local_offsets):
        if self.weight.size(0) == self.out_features:
            return self.weight.index_select(0, sampled_global_classes)
        return self.weight.index_select(0, sampled_local_offsets)

    def _set_margin_constants(self, m):
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def _positive_global_labels(self, label):
        if is_mixed_label(label):
            label1, label2, _ = label
            labels = torch.cat([label1.reshape(-1), label2.reshape(-1)]).long()
        else:
            labels = get_primary_label(label).reshape(-1).long()
        local_mask = (labels >= self.class_start) & (labels < self.class_end)
        return torch.unique(labels[local_mask], sorted=True)

    def _sample_global_classes(self, label, device):
        local_classes = torch.arange(
            self.class_start,
            self.class_end,
            device=device,
            dtype=torch.long,
        )
        positives = self._positive_global_labels(label).to(device)
        sample_count = max(
            int(self.local_out_features * self.sample_rate),
            int(positives.numel()),
            min(self.min_sample_classes, self.local_out_features),
        )
        sample_count = min(sample_count, self.local_out_features)

        if sample_count == self.local_out_features:
            return local_classes

        local_positive_offsets = positives - self.class_start
        negative_mask = torch.ones(
            self.local_out_features,
            device=device,
            dtype=torch.bool,
        )
        negative_mask[local_positive_offsets] = False
        negative_offsets = negative_mask.nonzero(as_tuple=False).flatten()
        negative_count = sample_count - positives.numel()
        negative_indices = torch.randperm(negative_offsets.numel(), device=device)[
            :negative_count
        ]
        sampled_offsets = torch.cat(
            [local_positive_offsets, negative_offsets[negative_indices]]
        )
        sampled_offsets = torch.sort(sampled_offsets).values
        return sampled_offsets + self.class_start

    def _sampled_one_hot(self, label, sampled_global_classes):
        one_hot = sampled_global_classes.new_zeros(
            (get_primary_label(label).numel(), sampled_global_classes.numel()),
            dtype=torch.float32,
        )

        def add_labels(labels, weight):
            positions = torch.searchsorted(sampled_global_classes, labels.long())
            valid = positions < sampled_global_classes.numel()
            valid = valid & (
                sampled_global_classes[
                    positions.clamp_max(sampled_global_classes.numel() - 1)
                ]
                == labels
            )
            if valid.any():
                rows = torch.arange(labels.numel(), device=labels.device)[valid]
                one_hot[rows, positions[valid]] += weight

        if is_mixed_label(label):
            label1, label2, lam = label
            add_labels(label1.reshape(-1).to(sampled_global_classes.device), lam)
            add_labels(label2.reshape(-1).to(sampled_global_classes.device), 1.0 - lam)
        else:
            add_labels(
                get_primary_label(label).reshape(-1).to(sampled_global_classes.device),
                1.0,
            )

        return one_hot.to(sampled_global_classes.device)

    def forward(self, x, label):
        self._sync_distributed_info()
        if self.world_size > 1 and is_mixed_label(label):
            raise ValueError(
                "DistributedPartialFCArcMarginProduct does not support mixup/cutmix "
                "labels across distributed classifier shards"
            )

        global_x, row_start, local_batch_size = _all_gather_equal_batch(x)
        global_label, _, _ = _all_gather_equal_labels(get_primary_label(label))

        sampled_global_classes = self._sample_global_classes(global_label, x.device)
        sampled_local_offsets = sampled_global_classes - self.class_start
        sampled_weight = self._select_sampled_weight(
            sampled_global_classes,
            sampled_local_offsets,
        )

        cosine = F.linear(F.normalize(global_x), F.normalize(sampled_weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = self._sampled_one_hot(global_label, sampled_global_classes).to(
            dtype=cosine.dtype
        )
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + (
                self.ls_eps / sampled_global_classes.numel()
            )

        sampled_output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        sampled_output *= self.s

        local_output = cosine.new_full(
            (global_x.size(0), self.local_out_features),
            self.unsampled_logit,
        )
        local_output[:, sampled_local_offsets] = sampled_output
        global_output = _all_gather_logits(
            local_output,
            self.out_features,
            self.unsampled_logit,
        )
        return global_output[row_start : row_start + local_batch_size]

    def update(self, m=0.5):
        if isinstance(m, dict):
            raise ValueError(
                "DistributedPartialFCArcMarginProduct does not support dynamic margins"
            )
        self.m = m
        self._set_margin_constants(m)
