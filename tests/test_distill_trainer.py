from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("hydra")
pytest.importorskip("omegaconf")
pytest.importorskip("albumentations")
edict = pytest.importorskip("easydict").EasyDict

from src.trainer import distill as distill_module


class StudentWithMargin(torch.nn.Module):
    def forward(self, images, targets=None):
        if targets is None:
            raise TypeError("targets are required for margin training")
        logits = torch.zeros(images.size(0), 2, device=images.device)
        return logits, images.float()


class EmbeddingStudent(torch.nn.Module):
    def forward(self, images):
        return images.float()


class Teacher(torch.nn.Module):
    def forward(self, images):
        return images.float()


def make_config(distill_loss_only):
    return SimpleNamespace(
        epochs=1,
        amp=False,
        debug=False,
        visualize_batch=False,
        loss=SimpleNamespace(kind="classification"),
        scheduler=SimpleNamespace(scheduler={}),
        train=SimpleNamespace(trainer=SimpleNamespace(grad_accum_steps=1)),
        distillation=SimpleNamespace(
            trainer=SimpleNamespace(
                kind="distill",
                params=SimpleNamespace(
                    distill_loss_only=distill_loss_only,
                    distill_loss_weight=1.0,
                ),
            )
        ),
        backbone=SimpleNamespace(norm_std=[1, 1, 1], norm_mean=[0, 0, 0]),
        transform=SimpleNamespace(
            cutmix=SimpleNamespace(p=0, alpha=0.5),
            mixup=SimpleNamespace(p=0, alpha=0.5),
        ),
    )


def patch_distill_dependencies(monkeypatch):
    def fake_get_loss(loss_config, device=None):
        if getattr(loss_config, "kind", None) == "distill":
            return {
                "mse": edict(
                    {
                        "loss_fn": torch.nn.MSELoss(),
                        "weight": 1.0,
                    }
                )
            }
        return {
            "ce": edict(
                {
                    "loss_fn": torch.nn.CrossEntropyLoss(),
                    "weight": 1.0,
                }
            )
        }

    monkeypatch.setattr(distill_module, "get_loss", fake_get_loss)
    monkeypatch.setattr(
        distill_module,
        "get_scheduler",
        lambda optimizer, scheduler_config: SimpleNamespace(step=lambda: None),
    )


def test_distill_valid_epoch_passes_targets_to_margin_student(monkeypatch):
    patch_distill_dependencies(monkeypatch)
    trainer = distill_module.DistillTrainer(
        config=make_config(distill_loss_only=False),
        model=StudentWithMargin(),
        model_teacher=Teacher(),
        optimizer=SimpleNamespace(),
        device="cpu",
    )
    loader = [
        (
            torch.ones(2, 3),
            torch.tensor([0, 1]),
            ["a.jpg", "b.jpg"],
        )
    ]

    stats = trainer.valid_epoch(loader)

    assert set(stats.losses) == {"mse", "ce", "total_loss"}


def test_distill_loss_only_valid_epoch_does_not_require_classification_loss(
    monkeypatch,
):
    patch_distill_dependencies(monkeypatch)
    trainer = distill_module.DistillTrainer(
        config=make_config(distill_loss_only=True),
        model=EmbeddingStudent(),
        model_teacher=Teacher(),
        optimizer=SimpleNamespace(),
        device="cpu",
    )
    loader = [
        (
            torch.ones(2, 3),
            torch.tensor([0, 1]),
            ["a.jpg", "b.jpg"],
        )
    ]

    stats = trainer.valid_epoch(loader)

    assert set(stats.losses) == {"mse", "total_loss"}
