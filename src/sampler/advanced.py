import collections

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from torch.utils.data.sampler import Sampler

from .m_per_class import get_labels_to_indices, safe_random_choice


class ClassBalancedSampler(WeightedRandomSampler):
    """
    Effective-number class-balanced sampler from Cui et al. 2019.
    """

    def __init__(
        self,
        labels,
        beta=0.9999,
        num_samples=None,
        replacement=True,
        seed=0,
    ):
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        labels = np.asarray(labels)
        counts = collections.Counter(labels.tolist())
        class_weights = {
            label: (1.0 - beta) / (1.0 - beta**count) for label, count in counts.items()
        }
        weights = torch.tensor(
            [class_weights[label] for label in labels.tolist()],
            dtype=torch.double,
        )
        if num_samples is None:
            num_samples = len(labels)
        self.seed = int(seed)
        self.epoch = 0
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)
        super(ClassBalancedSampler, self).__init__(
            weights=weights,
            num_samples=int(num_samples),
            replacement=replacement,
            generator=self.generator,
        )

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        self.generator.manual_seed(self.seed + self.epoch)

    def __iter__(self):
        self.generator.manual_seed(self.seed + self.epoch)
        self.epoch += 1
        return super().__iter__()


class HierarchicalPKSampler(Sampler):
    """
    Samples groups uniformly, then samples P labels and K examples per label.
    """

    def __init__(
        self,
        labels,
        groups=None,
        groups_per_batch=2,
        labels_per_group=2,
        samples_per_label=2,
        length_before_new_iter=100000,
        seed=0,
    ):
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        labels = np.asarray(labels)
        if groups is None:
            groups = labels
        if torch.is_tensor(groups):
            groups = groups.cpu().numpy()
        groups = np.asarray(groups)
        if labels.shape[0] != groups.shape[0]:
            raise ValueError("labels and groups must have the same length")

        self.groups_per_batch = int(groups_per_batch)
        self.labels_per_group = int(labels_per_group)
        self.samples_per_label = int(samples_per_label)
        self.batch_size = (
            self.groups_per_batch * self.labels_per_group * self.samples_per_label
        )
        self.list_size = int(length_before_new_iter)
        self.list_size -= self.list_size % self.batch_size
        self.seed = int(seed)
        self.epoch = 0

        self.group_to_labels = collections.defaultdict(list)
        self.label_to_indices = get_labels_to_indices(labels)
        for label, group in zip(labels.tolist(), groups.tolist()):
            if label not in self.group_to_labels[group]:
                self.group_to_labels[group].append(label)
        self.groups = np.asarray(list(self.group_to_labels.keys()))

        if len(self.groups) < self.groups_per_batch:
            raise ValueError("number of groups must be >= groups_per_batch")

    def __len__(self):
        return self.list_size

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        idx_list = [0] * self.list_size
        cursor = 0
        for _ in range(self.list_size // self.batch_size):
            batch_groups = safe_random_choice(
                self.groups, size=self.groups_per_batch, rng=rng
            )
            for group in batch_groups:
                labels = self.group_to_labels[group]
                batch_labels = safe_random_choice(
                    labels, size=self.labels_per_group, rng=rng
                )
                for label in batch_labels:
                    idx_list[cursor : cursor + self.samples_per_label] = (
                        safe_random_choice(
                            self.label_to_indices[label],
                            size=self.samples_per_label,
                            rng=rng,
                        )
                    )
                    cursor += self.samples_per_label
        self.epoch += 1
        return iter(idx_list)


class HardNegativeSampler(Sampler):
    """
    PK sampler that biases identities toward cached nearest negative classes.
    """

    def __init__(
        self,
        labels,
        embeddings,
        p=4,
        k=2,
        hard_negative_rate=0.5,
        length_before_new_iter=100000,
        seed=0,
    ):
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        labels = np.asarray(labels)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if labels.shape[0] != embeddings.shape[0]:
            raise ValueError("labels and embeddings must have the same length")

        self.p = int(p)
        self.k = int(k)
        self.batch_size = self.p * self.k
        self.hard_negative_rate = float(hard_negative_rate)
        self.list_size = int(length_before_new_iter)
        self.list_size -= self.list_size % self.batch_size
        self.seed = int(seed)
        self.epoch = 0
        self.labels_to_indices = get_labels_to_indices(labels)
        self.sample_labels = labels
        self.labels = np.asarray(list(self.labels_to_indices.keys()))
        if len(self.labels) < self.p:
            raise ValueError("number of unique labels must be >= p")
        self.hard_labels = self._build_hard_label_map(labels, embeddings)

    def _build_hard_label_map(self, labels, embeddings):
        cache_labels_to_indices = get_labels_to_indices(labels)
        centers = []
        center_labels = []
        for label in self.labels:
            if label not in cache_labels_to_indices:
                raise ValueError(f"Missing cached embeddings for label {label}")
            center_labels.append(label)
            centers.append(embeddings[cache_labels_to_indices[label]].mean(axis=0))
        centers = np.asarray(centers, dtype=np.float32)
        centers = centers / np.maximum(
            np.linalg.norm(centers, axis=1, keepdims=True), 1e-12
        )
        similarities = centers @ centers.T
        np.fill_diagonal(similarities, -np.inf)
        nearest = similarities.argmax(axis=1)
        return {
            center_labels[index]: center_labels[nearest_index]
            for index, nearest_index in enumerate(nearest)
        }

    def refresh(self, embeddings, labels=None):
        if labels is None:
            labels = self.sample_labels
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        if torch.is_tensor(embeddings):
            embeddings = embeddings.detach().cpu().numpy()
        labels = np.asarray(labels)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if labels.shape[0] != embeddings.shape[0]:
            raise ValueError("labels and embeddings must have the same length")
        self.hard_labels = self._build_hard_label_map(labels, embeddings)

    def __len__(self):
        return self.list_size

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        idx_list = [0] * self.list_size
        cursor = 0
        for _ in range(self.list_size // self.batch_size):
            seed_label = safe_random_choice(self.labels, size=1, rng=rng)[0]
            batch_labels = [seed_label]
            while len(batch_labels) < self.p:
                use_hard = rng.random() < self.hard_negative_rate
                if use_hard:
                    candidate = self.hard_labels[batch_labels[-1]]
                else:
                    candidate = safe_random_choice(self.labels, size=1, rng=rng)[0]
                if candidate not in batch_labels:
                    batch_labels.append(candidate)
            for label in batch_labels:
                idx_list[cursor : cursor + self.k] = safe_random_choice(
                    self.labels_to_indices[label], size=self.k, rng=rng
                )
                cursor += self.k
        self.epoch += 1
        return iter(idx_list)
