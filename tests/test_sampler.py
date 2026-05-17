from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from src.sampler import get_sampler
from src.sampler.m_per_class import MPerClassSampler


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
