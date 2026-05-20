from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from src.sampler import get_sampler
from src.sampler.advanced import (
    ClassBalancedSampler,
    HardNegativeSampler,
    HierarchicalPKSampler,
)
from src.sampler.m_per_class import MPerClassSampler
from src.sampler.pk import PKSampler


def test_balanced_sampler_is_returned_from_factory():
    sampler = get_sampler(
        labels=[0, 0, 1, 1, 2, 2],
        sampler_config=SimpleNamespace(
            type="balanced",
            m=2,
            batch_size=4,
            length_before_new_iter=8,
        ),
    )

    assert isinstance(sampler, MPerClassSampler)
    assert len(sampler) == 8


def test_default_sampler_factory_returns_none():
    sampler = get_sampler(labels=[0, 1], sampler_config=SimpleNamespace(type="default"))

    assert sampler is None


def test_pk_sampler_is_returned_from_factory():
    sampler = get_sampler(
        labels=[0, 0, 1, 1, 2, 2, 3, 3],
        sampler_config=SimpleNamespace(
            type="pk",
            p=2,
            k=2,
            batch_size=4,
            length_before_new_iter=8,
        ),
    )

    assert isinstance(sampler, PKSampler)
    assert len(sampler) == 8


def test_pk_sampler_returns_p_identities_with_k_samples_each():
    labels = [0, 0, 1, 1, 2, 2]
    sampler = PKSampler(labels=labels, p=2, k=2, batch_size=4, length_before_new_iter=4)

    batch_indices = list(iter(sampler))
    batch_labels = [labels[idx] for idx in batch_indices]
    counts = {label: batch_labels.count(label) for label in set(batch_labels)}

    assert len(batch_indices) == 4
    assert len(counts) == 2
    assert set(counts.values()) == {2}


def test_pk_sampler_is_epoch_seeded_and_reproducible():
    labels = [0, 0, 1, 1, 2, 2]
    sampler_a = PKSampler(
        labels=labels,
        p=2,
        k=2,
        batch_size=4,
        length_before_new_iter=8,
        seed=123,
    )
    sampler_b = PKSampler(
        labels=labels,
        p=2,
        k=2,
        batch_size=4,
        length_before_new_iter=8,
        seed=123,
    )

    sampler_a.set_epoch(4)
    sampler_b.set_epoch(4)

    assert list(iter(sampler_a)) == list(iter(sampler_b))


def test_class_balanced_sampler_is_returned_from_factory():
    sampler = get_sampler(
        labels=[0, 0, 0, 1],
        sampler_config=SimpleNamespace(
            type="class_balanced",
            beta=0.9,
            num_samples=4,
            replacement=True,
        ),
    )

    assert isinstance(sampler, ClassBalancedSampler)
    assert len(sampler) == 4


def test_hierarchical_pk_sampler_respects_groups_and_labels():
    labels = [0, 0, 1, 1, 2, 2, 3, 3]
    groups = ["a", "a", "a", "a", "b", "b", "b", "b"]
    sampler = HierarchicalPKSampler(
        labels=labels,
        groups=groups,
        groups_per_batch=2,
        labels_per_group=1,
        samples_per_label=2,
        length_before_new_iter=4,
    )

    batch_indices = list(iter(sampler))
    batch_groups = {groups[index] for index in batch_indices}

    assert len(batch_indices) == 4
    assert batch_groups == {"a", "b"}


def test_hard_negative_sampler_uses_embedding_cache():
    labels = [0, 0, 1, 1, 2, 2]
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
            [0.75, 0.25],
            [0.0, 1.0],
            [0.1, 0.9],
        ]
    )
    sampler = HardNegativeSampler(
        labels=labels,
        embeddings=embeddings,
        p=2,
        k=2,
        hard_negative_rate=1.0,
        length_before_new_iter=4,
    )

    batch_indices = list(iter(sampler))

    assert len(batch_indices) == 4
    assert len({labels[index] for index in batch_indices}) == 2


def test_hard_negative_sampler_rejects_too_few_labels():
    with pytest.raises(ValueError, match="unique labels"):
        HardNegativeSampler(
            labels=[0, 0],
            embeddings=torch.ones(2, 2),
            p=2,
            k=1,
            length_before_new_iter=2,
        )


def test_hard_negative_sampler_factory_loads_npz_cache(tmp_path):
    np = pytest.importorskip("numpy")
    cache_path = tmp_path / "hard_negative.npz"
    np.savez(
        cache_path,
        embeddings=np.asarray(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.8, 0.2],
                [0.75, 0.25],
                [0.0, 1.0],
                [0.1, 0.9],
            ],
            dtype=np.float32,
        ),
    )

    sampler = get_sampler(
        labels=[0, 0, 1, 1, 2, 2],
        sampler_config=SimpleNamespace(
            type="hard_negative",
            embeddings=None,
            embeddings_path=str(cache_path),
            p=2,
            k=2,
            hard_negative_rate=1.0,
            length_before_new_iter=4,
        ),
    )

    assert isinstance(sampler, HardNegativeSampler)
    assert len(sampler) == 4
