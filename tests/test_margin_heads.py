from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("omegaconf")
pytest.importorskip("hydra")
pytest.importorskip("albumentations")

from src.model.margin import get_margin
from src.model.margin.utils import build_one_hot, get_incremental_margin


def margin_config(margin_type, **kwargs):
    values = {
        "type": margin_type,
        "autoscale": False,
        "dynamic_margin": None,
        "id_counts": None,
        "s": 30.0,
        "m": 0.5,
        "m1": 1.0,
        "m3": 0.0,
        "h": 0.333,
        "t_alpha": 0.01,
        "use_batchnorm": False,
        "K": 3,
        "plus": margin_type.endswith("_plus"),
        "ls_eps": 0.1,
        "easy_margin": False,
    }
    values.update(kwargs)
    return SimpleNamespace(**values)


def test_softmax_margin_does_not_require_s_or_m_config_values():
    config = SimpleNamespace(
        type="softmax",
        autoscale=False,
        dynamic_margin=None,
        id_counts=None,
    )
    margin = get_margin(config, embeddings_size=4, n_classes=3)

    output = margin(torch.randn(2, 4), torch.tensor([0, 1]))

    assert output.shape == (2, 3)


def test_cosface_margin_runs_on_cpu_and_applies_label_smoothing():
    margin = get_margin(margin_config("cosface"), embeddings_size=4, n_classes=3)

    output = margin(torch.randn(2, 4), torch.tensor([0, 1]))

    assert output.shape == (2, 3)
    assert output.device.type == "cpu"


@pytest.mark.parametrize(
    "margin_type",
    [
        "adacos",
        "adaface",
        "arcface",
        "combined",
        "cosface",
        "curricularface",
        "elastic_arcface",
        "elastic_arcface_plus",
        "elastic_cosface",
        "elastic_cosface_plus",
        "softmax",
        "subcenter_arcface",
    ],
)
def test_all_registered_margin_heads_run_on_cpu(margin_type):
    config = (
        SimpleNamespace(
            type="softmax",
            autoscale=False,
            dynamic_margin=None,
            id_counts=None,
        )
        if margin_type == "softmax"
        else margin_config(margin_type)
    )
    margin = get_margin(config, embeddings_size=4, n_classes=3)

    output = margin(torch.randn(2, 4), torch.tensor([0, 1]))

    assert output.shape == (2, 3)
    assert output.device.type == "cpu"


def test_build_one_hot_supports_mixed_labels():
    labels = [
        torch.tensor([0, 1, 2]),
        torch.tensor([2, 2, 2]),
        0.25,
    ]

    one_hot = build_one_hot(labels, num_classes=3, device=torch.device("cpu"))

    assert torch.allclose(
        one_hot,
        torch.tensor(
            [
                [0.25, 0.0, 0.75],
                [0.0, 0.25, 0.75],
                [0.0, 0.0, 1.0],
            ]
        ),
    )


def test_incremental_margin_supports_dynamic_margin_dicts():
    margins = get_incremental_margin(
        m_max={0: 0.5, 1: 0.7},
        m_min=0.1,
        n_epochs=3,
        mode="linear",
    )

    assert len(margins) == 3
    assert margins[0] == {0: 0.1, 1: 0.1}
    assert margins[-1] == {0: 0.5, 1: 0.7}
