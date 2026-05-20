from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from src.config import ConfigValidationError, validate_training_config
from src.loss.pair import BatchHardTripletLoss, NTXentLoss, SupervisedContrastiveLoss
from src.loss.proxy import ProxyAnchorLoss, ProxyNCALoss


def make_training_config(loss_names, sampler_type):
    return SimpleNamespace(
        loss=SimpleNamespace(
            losses=[SimpleNamespace(name=loss_name) for loss_name in loss_names]
        ),
        dataloader=SimpleNamespace(sampler=SimpleNamespace(type=sampler_type)),
        train=SimpleNamespace(trainer=SimpleNamespace()),
    )


@pytest.mark.parametrize(
    "loss_fn",
    [
        BatchHardTripletLoss(margin=0.2),
        SupervisedContrastiveLoss(temperature=0.2),
        NTXentLoss(temperature=0.2),
    ],
)
def test_pair_losses_have_embedding_gradients(loss_fn):
    embeddings = torch.randn(6, 8, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])

    loss = loss_fn(embeddings, labels)
    loss.backward()

    assert loss.ndim == 0
    assert embeddings.grad is not None


def test_batch_hard_triplet_returns_zero_without_positive_pairs():
    embeddings = torch.randn(4, 8, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3])

    loss = BatchHardTripletLoss()(embeddings, labels)

    assert loss.item() == 0


@pytest.mark.parametrize(
    "loss_fn",
    [
        ProxyAnchorLoss(in_features=8, out_features=3),
        ProxyNCALoss(in_features=8, out_features=3),
    ],
)
def test_proxy_losses_have_embedding_and_proxy_gradients(loss_fn):
    embeddings = torch.randn(6, 8, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])

    loss = loss_fn(embeddings, labels)
    loss.backward()

    assert loss.ndim == 0
    assert embeddings.grad is not None
    assert loss_fn.proxies.grad is not None


def test_proxy_anchor_loss_is_stable_for_large_logits():
    loss_fn = ProxyAnchorLoss(in_features=2, out_features=2, margin=1.0, alpha=1000)
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    labels = torch.tensor([0, 1])
    with torch.no_grad():
        loss_fn.proxies.copy_(torch.tensor([[-1.0, 0.0], [0.0, -1.0]]))

    loss = loss_fn(embeddings, labels)

    assert torch.isfinite(loss)


@pytest.mark.parametrize("loss_name", ["proxy_anchor", "proxy_nca"])
def test_proxy_losses_do_not_require_positive_pair_sampler(loss_name):
    errors = validate_training_config(
        make_training_config([loss_name], sampler_type="default"),
        raise_on_error=False,
    )

    assert errors == []


def test_pair_batch_losses_reject_non_pair_sampler():
    with pytest.raises(ConfigValidationError, match="positive-pair sampler"):
        validate_training_config(
            make_training_config(["multi_similarity"], sampler_type="class_balanced")
        )
