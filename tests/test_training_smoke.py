from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
edict = pytest.importorskip("easydict").EasyDict

from src.loss.topofr import TopoFRLoss
from src.loss.transface import TransFaceEntropyMiningLoss
from src.trainer import base as base_trainer_module


class DummyKeypointModel(torch.nn.Module):
    def __init__(self):
        super(DummyKeypointModel, self).__init__()
        self.image_head = torch.nn.Linear(3 * 8 * 8, 4)
        self.keypoint_head = torch.nn.Linear(5 * 2, 4)
        self.classifier = torch.nn.Linear(4, 3)
        self.margin = SimpleNamespace(m=0.1)
        self.seen_keypoints = False

    def forward(self, images, labels, keypoints=None):
        del labels
        embeddings = self.image_head(images.flatten(1))
        if keypoints is not None:
            self.seen_keypoints = True
            embeddings = embeddings + self.keypoint_head(keypoints.flatten(1))
        return self.classifier(embeddings), embeddings


def trainer_config():
    return SimpleNamespace(
        epochs=1,
        amp=False,
        debug=False,
        visualize_batch=False,
        loss=SimpleNamespace(),
        scheduler=SimpleNamespace(scheduler={}),
        train=SimpleNamespace(trainer=SimpleNamespace(grad_accum_steps=1)),
        margin=SimpleNamespace(type="arcface", incremental_margin=None),
        transform=SimpleNamespace(
            cutmix=SimpleNamespace(p=0, alpha=0.5),
            mixup=SimpleNamespace(p=0, alpha=0.5),
            transface_dpap=SimpleNamespace(
                enabled=True,
                patch_grid=[2, 2],
                top_k=1,
                probability=1.0,
                alpha=1.0,
                ratio=1.0,
            ),
        ),
    )


def test_base_trainer_smoke_with_keypoints_dpap_and_image_embedding_loss(monkeypatch):
    monkeypatch.setattr(
        base_trainer_module,
        "get_loss",
        lambda loss_config, device=None: {
            "ehsm": edict(
                {
                    "loss_fn": TransFaceEntropyMiningLoss(gamma=1.0),
                    "weight": 1.0,
                    "input": "output",
                    "mixable": True,
                }
            ),
            "topofr": edict(
                {
                    "loss_fn": TopoFRLoss(max_samples=4),
                    "weight": 0.01,
                    "input": "images_embeddings",
                    "mixable": False,
                }
            ),
        },
    )
    monkeypatch.setattr(
        base_trainer_module,
        "get_scheduler",
        lambda optimizer, scheduler_config: SimpleNamespace(step=lambda *args: None),
    )
    monkeypatch.setattr(
        base_trainer_module,
        "get_warmup_scheduler",
        lambda optimizer, scheduler_config: None,
    )

    model = DummyKeypointModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    initial_weight = model.image_head.weight.detach().clone()
    images = torch.randn(4, 3, 8, 8)
    targets = {
        "label": torch.tensor([0, 1, 2, 1]),
        "keypoints": torch.rand(4, 5, 2),
    }
    loader = [(images, targets, ["a.jpg", "b.jpg", "c.jpg", "d.jpg"])]

    trainer = base_trainer_module.BaseTrainer(
        config=trainer_config(),
        model=model,
        optimizer=optimizer,
        device="cpu",
    )
    stats = trainer.train_epoch(loader)

    assert model.seen_keypoints is True
    assert set(["ehsm", "topofr", "total_loss"]).issubset(stats.losses)
    assert not torch.allclose(initial_weight, model.image_head.weight.detach())
