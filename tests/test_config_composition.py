from pathlib import Path

import pytest

pytest.importorskip("hydra")
pytest.importorskip("omegaconf")
pytest.importorskip("torch")

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.loss import get_loss

CONFIG_DIR = str((Path(__file__).resolve().parents[1] / "configs").resolve())


def compose_train_config(overrides):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config_train", overrides=overrides)
    OmegaConf.resolve(cfg)
    return cfg


@pytest.mark.parametrize(
    "override",
    [
        "loss=circle",
        "loss=uniface_uce",
        "loss=topofr",
        "loss=topofr_sde",
        "loss=transface_ehsm",
        "loss=multi_similarity",
        "loss=batch_hard_triplet",
        "loss=supcon",
        "loss=ntxent",
        "loss=proxy_anchor",
        "loss=proxy_nca",
        "loss=sface",
        "loss=boundaryface",
        "loss=unitsface",
        "loss=focal_loss",
        "loss=soft_cross_entropy",
        "margin=magface",
        "margin=circle",
        "margin=x2_softmax",
        "margin=qamface",
        "margin=sphereface",
        "margin=partialfc_arcface",
        "margin=distributed_partialfc_arcface",
        "dataloader=pk",
        "dataloader=class_balanced",
        "dataloader=hierarchical_pk",
        "dataloader=hard_negative",
        "optimizer=lamb",
        "optimizer=muon",
        "optimizer=schedule_free_adamw",
        "backbone=edgeface_base",
        "backbone=edgeface_s_gamma_05",
        "backbone=edgeface_xs_gamma_06",
        "backbone=edgeface_xxs",
        "backbone=mobilefacenet",
        "backbone=am_radio_v2_5_b",
        "backbone=kprpe_vit_base_patch16_224",
        "backbone=iresnet50",
        "backbone=dinov2_vit_b14",
        "backbone=siglip_vit_b16",
        "backbone=openclip_mobileclip_s0",
        "evaluation/evaluator=face_verification",
        "evaluation/evaluator=ijb_template",
        "evaluation/knn=faiss_ivf",
        "evaluation/knn=faiss_hnsw",
        "evaluation.knn.rerank.enabled=True",
        "train.trainer.xbm.enabled=True",
        "transform=default",
        "transform=randaugment",
        "transform=trivialaugment",
        "transform=augmix",
    ],
)
def test_new_config_overrides_compose_and_resolve(override):
    cfg = compose_train_config([override, "n_classes=7", "embeddings_size=4"])

    assert cfg is not None


@pytest.mark.parametrize(
    "loss_name",
    [
        "circle",
        "uniface_uce",
        "topofr",
        "topofr_sde",
        "transface_ehsm",
        "multi_similarity",
        "batch_hard_triplet",
        "supcon",
        "ntxent",
        "proxy_anchor",
        "proxy_nca",
        "sface",
        "boundaryface",
        "unitsface",
        "focal_loss",
        "soft_cross_entropy",
    ],
)
def test_new_loss_configs_instantiate(loss_name):
    cfg = compose_train_config(
        [
            f"loss={loss_name}",
            "n_classes=7",
            "embeddings_size=4",
        ]
    )

    loss_fns = get_loss(cfg.loss, device="cpu")

    assert loss_fns
    assert all(
        hasattr(loss_params.loss_fn, "forward") for loss_params in loss_fns.values()
    )
