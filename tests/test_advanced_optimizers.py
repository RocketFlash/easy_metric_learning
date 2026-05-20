from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from src.optimizer.lamb import LAMB
from src.optimizer.muon import Muon, zeropower_via_newtonschulz
from src.optimizer.schedule_free_adamw import ScheduleFreeAdamW
from src.optimizer.sophia_g import SophiaG
from src.trainer.compile import maybe_compile_model
from src.trainer.fsdp import is_fsdp_enabled, maybe_wrap_fsdp


@pytest.mark.parametrize(
    "optimizer_cls",
    [
        LAMB,
        Muon,
        ScheduleFreeAdamW,
    ],
)
def test_advanced_optimizers_update_parameters(optimizer_cls):
    model = torch.nn.Linear(4, 2)
    optimizer = optimizer_cls(model.parameters(), lr=1e-2)
    before = model.weight.detach().clone()

    loss = model(torch.randn(3, 4)).pow(2).mean()
    loss.backward()
    optimizer.step()

    assert not torch.allclose(before, model.weight)


def test_schedule_free_adamw_swaps_eval_average():
    model = torch.nn.Linear(4, 2)
    optimizer = ScheduleFreeAdamW(model.parameters(), lr=1e-2)
    loss = model(torch.randn(3, 4)).pow(2).mean()
    loss.backward()
    optimizer.step()

    train_weight = model.weight.detach().clone()
    optimizer.eval()
    eval_weight = model.weight.detach().clone()
    optimizer.train()

    assert not torch.allclose(train_weight, eval_weight)
    assert torch.allclose(model.weight, train_weight)


def test_schedule_free_adamw_state_dict_restores_train_weights_from_eval_mode():
    model = torch.nn.Linear(4, 2)
    optimizer = ScheduleFreeAdamW(model.parameters(), lr=1e-2)
    loss = model(torch.randn(3, 4)).pow(2).mean()
    loss.backward()
    optimizer.step()
    train_weight = model.weight.detach().clone()

    optimizer.eval()
    optimizer.state_dict()

    assert torch.allclose(model.weight, train_weight)


def test_sophiag_uses_batch_size_from_param_group():
    model = torch.nn.Linear(4, 2)
    optimizer = SophiaG(model.parameters(), lr=1e-2, bs=32)

    assert optimizer.param_groups[0]["bs"] == 32


def test_muon_newton_schulz_uses_paper_coefficients():
    matrix = torch.tensor([[1.0, 0.2], [0.3, 0.7]])
    normalized = matrix / (matrix.norm() + 1e-7)
    gram = normalized @ normalized.t()
    expected = (
        3.4445 * normalized + (-4.7750 * gram + 2.0315 * gram @ gram) @ normalized
    )

    output = zeropower_via_newtonschulz(matrix, steps=1)

    assert torch.allclose(output, expected)


def test_fsdp_disabled_returns_original_model():
    model = torch.nn.Linear(2, 2)
    config = SimpleNamespace(
        train=SimpleNamespace(
            trainer=SimpleNamespace(fsdp=SimpleNamespace(enabled=False))
        )
    )

    assert is_fsdp_enabled(config) is False
    assert maybe_wrap_fsdp(model, config) is model


def test_compile_disabled_returns_original_model():
    model = torch.nn.Linear(2, 2)
    config = SimpleNamespace(
        train=SimpleNamespace(
            trainer=SimpleNamespace(compile=SimpleNamespace(enabled=False))
        )
    )

    assert maybe_compile_model(model, config) is model
