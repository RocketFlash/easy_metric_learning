import numpy as np
import torch
from torch.utils.data.sampler import Sampler

from .m_per_class import get_labels_to_indices, safe_random_choice


class PKSampler(Sampler):
    """
    Samples P identities with K examples each per batch.

    This is the standard batch construction for pair/proxy metric losses where
    each batch needs positive pairs and a diverse set of negatives.
    """

    def __init__(
        self,
        labels,
        p,
        k,
        batch_size=None,
        length_before_new_iter=100000,
        seed=0,
    ):
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        self.p = int(p)
        self.k = int(k)
        self.batch_size = int(batch_size) if batch_size is not None else self.p * self.k
        self.labels_to_indices = get_labels_to_indices(labels)
        self.labels = np.array(list(self.labels_to_indices.keys()))
        self.list_size = int(length_before_new_iter)
        self.seed = int(seed)
        self.epoch = 0

        assert self.p > 0, "p must be positive"
        assert self.k > 0, "k must be positive"
        assert self.batch_size == self.p * self.k, "batch_size must equal p * k"
        assert len(self.labels) >= self.p, "number of unique labels must be >= p"
        assert self.list_size >= self.batch_size

        self.list_size -= self.list_size % self.batch_size

    def __len__(self):
        return self.list_size

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        idx_list = [0] * self.list_size
        i = 0
        num_iters = self.list_size // self.batch_size

        for _ in range(num_iters):
            label_set = safe_random_choice(self.labels, size=self.p, rng=rng)
            for label in label_set:
                indices = self.labels_to_indices[label]
                idx_list[i : i + self.k] = safe_random_choice(
                    indices, size=self.k, rng=rng
                )
                i += self.k

        self.epoch += 1
        return iter(idx_list)
