import pytest

torch = pytest.importorskip("torch")

from src.loss.multi_similarity import MultiSimilarityLoss


def test_multi_similarity_loss_has_gradients():
    loss_fn = MultiSimilarityLoss(alpha=2.0, beta=20.0, base=0.5)
    embeddings = torch.randn(6, 8, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])

    loss = loss_fn(embeddings, labels)
    loss.backward()

    assert loss.ndim == 0
    assert embeddings.grad is not None


def test_multi_similarity_loss_returns_zero_without_positive_pairs():
    loss_fn = MultiSimilarityLoss(alpha=2.0, beta=20.0, base=0.5)
    embeddings = torch.randn(4, 8, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3])

    loss = loss_fn(embeddings, labels)

    assert loss.item() == 0


def test_multi_similarity_loss_accepts_tuple_labels():
    loss_fn = MultiSimilarityLoss(alpha=2.0, beta=20.0, base=0.5)
    embeddings = torch.randn(4, 8, requires_grad=True)
    labels = (torch.tensor([0, 0, 1, 1]), torch.tensor([1, 1, 0, 0]))

    loss = loss_fn(embeddings, labels)

    assert loss.ndim == 0
