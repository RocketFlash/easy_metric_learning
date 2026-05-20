import pytest

torch = pytest.importorskip("torch")

from src.loss.transface import TransFaceEntropyMiningLoss
from src.trainer.dpap import apply_dpap_transform, get_dpap_transform
from src.transform.transface import DynamicPatchAmplitudeMix


def test_transface_entropy_mining_loss_has_gradients():
    loss_fn = TransFaceEntropyMiningLoss(gamma=1.0, hard_fraction=0.5)
    logits = torch.randn(6, 4, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3, 1, 2])

    loss = loss_fn(logits, labels)
    loss.backward()

    assert loss.ndim == 0
    assert logits.grad is not None


def test_dynamic_patch_amplitude_mix_preserves_shape_and_changes_selected_patch():
    torch.manual_seed(7)
    transform = DynamicPatchAmplitudeMix(
        patch_grid=(2, 2),
        top_k=1,
        probability=1.0,
        alpha=1.0,
    )
    images = torch.zeros(2, 3, 8, 8)
    images[1] = 1.0
    patch_scores = torch.tensor(
        [
            [10.0, 0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0, 0.0],
        ]
    )

    mixed = transform(images, patch_scores)

    assert mixed.shape == images.shape
    assert not torch.allclose(mixed[:, :, :4, :4], images[:, :, :4, :4])


def test_dpap_trainer_helper_is_disabled_by_default():
    config = type("Config", (), {"transform": type("Transform", (), {})()})()
    images = torch.randn(2, 3, 8, 8)

    transform = get_dpap_transform(config)

    assert transform is None
    assert apply_dpap_transform(transform, images) is images
