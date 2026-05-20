from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from src.config import ConfigValidationError, validate_training_config
from src.trainer.loss_inputs import calculate_weighted_loss
from src.trainer.xbm import CrossBatchMemory


class LossParams:
    def __init__(self, loss_fn, input_key="embeddings", xbm=True):
        self.loss_fn = loss_fn
        self.input = input_key
        self.weight = 1.0
        self.xbm = xbm


def test_cross_batch_memory_keeps_recent_embeddings_in_queue_order():
    xbm = CrossBatchMemory(memory_size=3, embedding_size=2, device="cpu")

    xbm.enqueue(
        torch.tensor([[1.0, 0.0], [2.0, 0.0]]),
        torch.tensor([0, 1]),
    )
    xbm.enqueue(
        torch.tensor([[3.0, 0.0], [4.0, 0.0]]),
        torch.tensor([2, 3]),
    )

    memory_embeddings, memory_labels = xbm.get(torch.zeros(1, 2))

    assert len(xbm) == 3
    assert memory_embeddings[:, 0].tolist() == [2.0, 3.0, 4.0]
    assert memory_labels.tolist() == [1, 2, 3]


def test_weighted_embedding_loss_extends_inputs_with_xbm_memory():
    xbm = CrossBatchMemory(memory_size=4, embedding_size=2, device="cpu")
    xbm.enqueue(torch.ones(2, 2) * 2, torch.tensor([1, 1]))
    observed = {}

    def loss_fn(embeddings, labels):
        observed["shape"] = tuple(embeddings.shape)
        observed["labels"] = labels.tolist()
        return embeddings[:, 0].sum()

    embeddings = torch.ones(1, 2, requires_grad=True)
    loss = calculate_weighted_loss(
        LossParams(loss_fn),
        output=torch.zeros(1, 2),
        embeddings=embeddings,
        targets=torch.tensor([0]),
        xbm=xbm,
    )

    loss.backward()

    assert observed["shape"] == (3, 2)
    assert observed["labels"] == [0, 1, 1]
    assert embeddings.grad is not None


def test_xbm_validation_rejects_invalid_memory_size():
    config = SimpleNamespace(
        train=SimpleNamespace(
            trainer=SimpleNamespace(
                xbm=SimpleNamespace(enabled=True, memory_size=0),
            )
        )
    )

    with pytest.raises(ConfigValidationError, match="xbm.memory_size"):
        validate_training_config(config)
