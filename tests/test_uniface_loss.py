import pytest

torch = pytest.importorskip("torch")

from src.loss.uniface import UnifiedCrossEntropyLoss


def test_uniface_uce_has_gradients():
    loss_fn = UnifiedCrossEntropyLoss(
        in_features=4,
        out_features=6,
        sample_rate=1.0,
    )
    embeddings = torch.randn(5, 4, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3, 4])

    loss = loss_fn(embeddings, labels)
    loss.backward()

    assert loss.ndim == 0
    assert embeddings.grad is not None
    assert loss_fn.weight.grad is not None


def test_uniface_sampling_keeps_positive_classes():
    loss_fn = UnifiedCrossEntropyLoss(
        in_features=4,
        out_features=20,
        sample_rate=0.2,
        min_sample_classes=4,
    )
    loss_fn.train()

    embeddings = torch.randn(3, 4, requires_grad=True)
    labels = torch.tensor([2, 7, 11])

    loss = loss_fn(embeddings, labels)

    sampled = set(loss_fn.last_sampled_indices.tolist())
    assert loss.ndim == 0
    assert {2, 7, 11}.issubset(sampled)
    assert len(sampled) >= 4
