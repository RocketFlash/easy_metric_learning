import pytest

torch = pytest.importorskip("torch")

from src.loss.focal import FocalLoss
from src.loss.soft_cross_entropy import SoftCrossEntropyLoss


def test_focal_loss_applies_per_sample_weighting():
    logits = torch.tensor(
        [
            [5.0, 0.0],
            [0.1, 0.0],
        ]
    )
    targets = torch.tensor([0, 1])
    loss_fn = FocalLoss(gamma=2.0)

    logp = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
    p = torch.exp(-logp)
    expected = (((1 - p) ** 2.0) * logp).mean()

    assert torch.allclose(loss_fn(logits, targets), expected)


def test_soft_cross_entropy_constructs_without_weight():
    loss_fn = SoftCrossEntropyLoss(label_smoothing=0.1, num_classes=3)
    logits = torch.randn(2, 3)
    targets = torch.tensor([0, 2])

    loss = loss_fn(logits, targets)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
