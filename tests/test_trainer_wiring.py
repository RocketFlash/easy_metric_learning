from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
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
            "base": edict(
                {
                    "loss_fn": base_loss,
                    "weight": 2.0,
                    "input": "embeddings",
                    "mixable": False,
                }
            )
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
    assert loss_params.input == "embeddings"
    assert loss_params.mixable is False
    assert loss_params.loss_fn(None, 25.0) == 25.0


def test_base_trainer_steps_scheduler_inside_warmup_dampening(monkeypatch):
    class FakeWarmup:
        def __init__(self):
            self.active = False

        def dampening(self):
            warmup = self

            class Context:
                def __enter__(self):
                    warmup.active = True

                def __exit__(self, exc_type, exc, tb):
                    warmup.active = False

            return Context()

    warmup = FakeWarmup()
    calls = []

    monkeypatch.setattr(
        base_trainer_module,
        "get_loss",
        lambda loss_config, device=None: {},
    )
    monkeypatch.setattr(
        base_trainer_module,
        "get_scheduler",
        lambda optimizer, scheduler_config: SimpleNamespace(),
    )
    monkeypatch.setattr(
        base_trainer_module,
        "get_warmup_scheduler",
        lambda optimizer, scheduler_config: warmup,
    )
    monkeypatch.setattr(
        base_trainer_module,
        "step_scheduler",
        lambda scheduler, metric=None: calls.append((warmup.active, metric)),
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
    trainer = base_trainer_module.BaseTrainer(
        config=config,
        model=SimpleNamespace(margin=SimpleNamespace(m=0.5)),
        optimizer=SimpleNamespace(),
        device="cpu",
    )

    trainer._step_scheduler(metric=0.5)

    assert calls == [(True, 0.5)]


def test_base_trainer_sets_cosine_tmax_to_remaining_epoch_count(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        base_trainer_module,
        "get_loss",
        lambda loss_config, device=None: {},
    )

    def fake_get_scheduler(optimizer, scheduler_config):
        captured["T_max"] = scheduler_config.scheduler["T_max"]
        return SimpleNamespace(step=lambda: None)

    monkeypatch.setattr(base_trainer_module, "get_scheduler", fake_get_scheduler)
    monkeypatch.setattr(
        base_trainer_module,
        "get_warmup_scheduler",
        lambda optimizer, scheduler_config: None,
    )

    config = SimpleNamespace(
        epochs=5,
        amp=False,
        debug=False,
        visualize_batch=False,
        loss=SimpleNamespace(),
        scheduler=SimpleNamespace(scheduler={"T_max": 5}),
        train=SimpleNamespace(trainer=SimpleNamespace(grad_accum_steps=1)),
        margin=SimpleNamespace(type="arcface", incremental_margin=None),
    )

    base_trainer_module.BaseTrainer(
        config=config,
        model=SimpleNamespace(margin=SimpleNamespace(m=0.5)),
        optimizer=SimpleNamespace(),
        device="cpu",
        epoch=2,
    )

    assert captured["T_max"] == 4


def test_base_trainer_formats_tuple_margin_for_progress_display(monkeypatch):
    monkeypatch.setattr(
        base_trainer_module,
        "get_loss",
        lambda loss_config, device=None: {},
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
        margin=SimpleNamespace(type="magface", incremental_margin=None),
    )
    trainer = base_trainer_module.BaseTrainer(
        config=config,
        model=SimpleNamespace(margin=SimpleNamespace(m=(0.45, 0.8))),
        optimizer=SimpleNamespace(),
        device="cpu",
    )

    assert trainer._get_margin_value() == "0.45..0.8"


def test_base_trainer_updates_ema_model_after_optimizer_step(monkeypatch, tmp_path):
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super(TinyModel, self).__init__()
            self.linear = torch.nn.Linear(2, 2)

        def forward(self, images, targets):
            embeddings = self.linear(images)
            return embeddings, embeddings

    def loss_fn(preds, targets):
        return preds.pow(2).mean()

    monkeypatch.setattr(
        base_trainer_module,
        "get_loss",
        lambda loss_config, device=None: {
            "base": edict(
                {
                    "loss_fn": loss_fn,
                    "weight": 1.0,
                    "input": "output",
                    "mixable": True,
                }
            )
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
        transform=SimpleNamespace(
            cutmix=SimpleNamespace(p=0.0, alpha=1.0),
            mixup=SimpleNamespace(p=0.0, alpha=1.0),
        ),
        train=SimpleNamespace(
            trainer=SimpleNamespace(
                grad_accum_steps=1,
                model_averaging=SimpleNamespace(
                    enabled=True,
                    type="ema",
                    decay=0.9,
                    start_epoch=1,
                    use_for_eval=True,
                ),
            )
        ),
        margin=SimpleNamespace(type="arcface", incremental_margin=None),
    )
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = base_trainer_module.BaseTrainer(
        config=config,
        model=model,
        optimizer=optimizer,
        device="cpu",
        work_dir=tmp_path,
    )
    train_loader = [
        (torch.ones(2, 2), torch.tensor([0, 1]), ["a.jpg", "b.jpg"]),
    ]

    trainer.train_epoch(train_loader)

    assert trainer.averaged_model is not None
    assert int(trainer.averaged_model.n_averaged.item()) == 1
    assert trainer.get_eval_model() is trainer.averaged_model.module
