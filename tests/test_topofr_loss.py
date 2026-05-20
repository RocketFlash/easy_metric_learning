from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from src.loss.topofr import TopoFRLoss, TopoFRSDECrossEntropyLoss
from src.trainer.loss_inputs import calculate_weighted_loss


def test_topofr_loss_has_gradients():
    loss_fn = TopoFRLoss(max_samples=8)
    images = torch.randn(6, 3, 8, 8)
    embeddings = torch.randn(6, 4, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])

    loss = loss_fn(images, embeddings, labels)
    loss.backward()

    assert loss.ndim == 0
    assert embeddings.grad is not None


def test_weighted_loss_routes_images_and_embeddings():
    def loss_fn(images, embeddings, targets):
        return images.sum() + embeddings.sum() + targets.sum()

    loss_params = SimpleNamespace(
        loss_fn=loss_fn,
        input="images_embeddings",
        weight=2.0,
    )

    loss = calculate_weighted_loss(
        loss_params,
        output=torch.zeros(2, 3),
        embeddings=torch.ones(2, 3),
        targets=torch.tensor([1, 2]),
        images=torch.ones(2, 3, 2, 2),
    )

    assert loss.item() == 66


def test_topofr_sde_cross_entropy_has_gradients():
    loss_fn = TopoFRSDECrossEntropyLoss(gum_iters=5)
    logits = torch.randn(6, 4, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3, 1, 2])

    loss = loss_fn(logits, labels)
    loss.backward()

    assert loss.ndim == 0
    assert logits.grad is not None
