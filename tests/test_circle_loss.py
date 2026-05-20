import pytest

torch = pytest.importorskip("torch")

from src.loss.circle import CircleLoss
from src.trainer.loss_inputs import calculate_weighted_loss


class LossParams:
    def __init__(self, loss_fn, input_key):
        self.loss_fn = loss_fn
        self.input = input_key
        self.weight = 1.0


def test_pair_circle_loss_has_gradients():
    loss_fn = CircleLoss(m=0.25, gamma=32)
    embeddings = torch.randn(4, 8, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1])

    loss = loss_fn(embeddings, labels)
    loss.backward()

    assert loss.ndim == 0
    assert embeddings.grad is not None


def test_pair_circle_loss_returns_zero_without_positive_pairs():
    loss_fn = CircleLoss(m=0.25, gamma=32)
    embeddings = torch.randn(4, 8, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3])

    loss = loss_fn(embeddings, labels)

    assert loss.item() == 0


def test_embedding_loss_input_selects_embeddings():
    def loss_fn(preds, targets):
        return preds.sum() + targets.sum()

    loss = calculate_weighted_loss(
        LossParams(loss_fn, "embeddings"),
        output=torch.ones(2, 3),
        embeddings=torch.ones(2, 3) * 2,
        targets=torch.tensor([0, 1]),
    )

    assert loss.item() == 13
