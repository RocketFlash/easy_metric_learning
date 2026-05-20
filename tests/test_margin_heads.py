from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("omegaconf")
pytest.importorskip("hydra")
pytest.importorskip("albumentations")

from src.model.margin import get_margin
from src.model.margin.elasticface import _build_elastic_margin_hot
from src.model.margin.utils import build_one_hot, get_incremental_margin


def margin_config(margin_type, **kwargs):
    values = {
        "type": margin_type,
        "autoscale": False,
        "dynamic_margin": None,
        "id_counts": None,
        "s": 30.0,
        "m": 0.5,
        "min_m": 0.05,
        "m1": 1.0,
        "m3": 0.0,
        "h": 0.333,
        "t_alpha": 0.01,
        "use_batchnorm": False,
        "l_a": 10.0,
        "u_a": 110.0,
        "l_margin": 0.45,
        "u_margin": 0.8,
        "lambda_g": 35.0,
        "min_s": 16.0,
        "sample_rate": 0.5,
        "min_sample_classes": 2,
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
        "circle",
        "combined",
        "cosface",
        "curricularface",
        "elastic_arcface",
        "elastic_arcface_plus",
        "elastic_cosface",
        "elastic_cosface_plus",
        "magface",
        "distributed_partialfc_arcface",
        "partialfc_arcface",
        "qamface",
        "softmax",
        "sphereface",
        "subcenter_arcface",
        "x2_softmax",
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


@pytest.mark.parametrize(
    "margin_type",
    [
        "elastic_arcface",
        "elastic_arcface_plus",
        "elastic_cosface",
        "elastic_cosface_plus",
    ],
)
def test_elasticface_margins_accept_mixed_labels(margin_type):
    margin = get_margin(margin_config(margin_type), embeddings_size=4, n_classes=3)
    labels = [
        torch.tensor([0, 1]),
        torch.tensor([1, 2]),
        0.25,
    ]

    output = margin(torch.randn(2, 4), labels)

    assert output.shape == (2, 3)


def test_curricularface_accepts_mixed_labels_without_nans():
    margin = get_margin(margin_config("curricularface"), embeddings_size=4, n_classes=3)
    labels = [
        torch.tensor([0, 1]),
        torch.tensor([1, 2]),
        0.25,
    ]
    embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
        ]
    )

    output = margin(embeddings, labels)

    assert output.shape == (2, 3)
    assert torch.isfinite(output).all()


def test_elasticface_plus_assigns_largest_margin_to_hardest_samples(monkeypatch):
    monkeypatch.setattr(
        torch,
        "normal",
        lambda mean, std, size, device: torch.tensor(
            [[0.1], [0.2], [0.3]], device=device
        ),
    )
    cos_theta = torch.tensor(
        [
            [0.9, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.5],
        ]
    )
    labels = torch.tensor([0, 1, 2])

    margin_hot = _build_elastic_margin_hot(
        labels,
        cos_theta,
        margin_value=0.2,
        std=0.01,
        plus=True,
    )

    assert margin_hot[0, 0] == pytest.approx(0.1)
    assert margin_hot[1, 1] == pytest.approx(0.3)
    assert margin_hot[2, 2] == pytest.approx(0.2)


def test_combined_margin_uses_m1_multiplier():
    margin = get_margin(
        margin_config("combined", s=1.0, m1=2.0, m=0.1, m3=0.0, ls_eps=0.0),
        embeddings_size=2,
        n_classes=2,
    )
    with torch.no_grad():
        margin.weight.copy_(torch.eye(2))
    embeddings = torch.tensor([[2**-0.5, 2**-0.5]])

    output = margin(embeddings, torch.tensor([0]))

    expected_target = torch.cos(torch.tensor(2.0 * torch.pi / 4.0 + 0.1))
    assert float(output[0, 0].detach()) == pytest.approx(
        float(expected_target), abs=1e-5
    )


def test_adacos_scale_is_persistent_buffer():
    margin = get_margin(margin_config("adacos"), embeddings_size=4, n_classes=3)

    assert "s" in margin.state_dict()
    assert torch.is_tensor(margin.s)


def test_magface_exposes_regularization_loss_with_gradients():
    margin = get_margin(
        margin_config(
            "magface",
            s=64.0,
            l_a=1.0,
            u_a=10.0,
            l_margin=0.2,
            u_margin=0.4,
            lambda_g=2.0,
        ),
        embeddings_size=4,
        n_classes=3,
    )
    embeddings = torch.randn(2, 4, requires_grad=True)

    output = margin(embeddings, torch.tensor([0, 1]))
    regularization_loss = margin.regularization_loss()

    assert output.shape == (2, 3)
    assert regularization_loss.ndim == 0
    (output.mean() + regularization_loss).backward()
    assert embeddings.grad is not None


def test_circle_margin_update_changes_margin_value():
    margin = get_margin(
        margin_config("circle", s=32.0, m=0.1),
        embeddings_size=4,
        n_classes=3,
    )

    margin.update(0.2)
    output = margin(torch.randn(2, 4), torch.tensor([0, 1]))

    assert margin.m == 0.2
    assert output.shape == (2, 3)


def test_partialfc_arcface_keeps_full_class_output_and_samples_negatives():
    margin = get_margin(
        margin_config(
            "partialfc_arcface",
            s=16.0,
            m=0.2,
            sample_rate=0.5,
            min_sample_classes=4,
        ),
        embeddings_size=4,
        n_classes=10,
    )

    output = margin(torch.randn(3, 4), torch.tensor([0, 1, 2]))

    assert output.shape == (3, 10)
    assert (output[:, [0, 1, 2]] > margin.unsampled_logit).all()
    assert (output == margin.unsampled_logit).any()


def test_distributed_partialfc_arcface_runs_without_distributed_init():
    margin = get_margin(
        margin_config(
            "distributed_partialfc_arcface",
            s=16.0,
            m=0.2,
            sample_rate=0.5,
            min_sample_classes=4,
        ),
        embeddings_size=4,
        n_classes=10,
    )

    output = margin(torch.randn(3, 4), torch.tensor([0, 1, 2]))

    assert output.shape == (3, 10)
    assert margin.local_out_features == 10


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
