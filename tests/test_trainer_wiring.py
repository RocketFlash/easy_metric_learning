from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("hydra")
pytest.importorskip("omegaconf")
pytest.importorskip("albumentations")
edict = pytest.importorskip("easydict").EasyDict

from src.trainer import base as base_trainer_module


def test_base_trainer_wraps_mix_losses_without_dropping_weights(monkeypatch):
    def base_loss(preds, targets):
        return targets

    monkeypatch.setattr(
        base_trainer_module,
        "get_loss",
        lambda loss_config, device=None: {
            "base": edict({"loss_fn": base_loss, "weight": 2.0})
        },
    )
    monkeypatch.setattr(
        base_trainer_module,
        "get_scheduler",
        lambda optimizer, scheduler_config: SimpleNamespace(step=lambda: None),
    )
    monkeypatch.setattr(
        base_trainer_module,
        "get_warmup_scheduler",
        lambda optimizer, scheduler_config: None,
    )

    config = SimpleNamespace(
        epochs=1,
        amp=False,
        debug=False,
        visualize_batch=False,
        loss=SimpleNamespace(),
        scheduler=SimpleNamespace(scheduler={}),
        train=SimpleNamespace(trainer=SimpleNamespace(grad_accum_steps=1)),
        margin=SimpleNamespace(type="arcface", incremental_margin=None),
    )
    model = SimpleNamespace(margin=SimpleNamespace(m=0.5))

    trainer = base_trainer_module.BaseTrainer(
        config=config,
        model=model,
        optimizer=SimpleNamespace(),
        device="cpu",
    )

    loss_params = trainer.mix_loss_fns["base"]
    assert loss_params.weight == 2.0
    assert loss_params.loss_fn(None, (10.0, 30.0, 0.25)) == 25.0
