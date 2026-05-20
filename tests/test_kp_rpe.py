import pytest

torch = pytest.importorskip("torch")

from src.model.modules.kp_rpe import KeypointRelativePositionBias, build_patch_grid


def test_build_patch_grid_returns_normalized_coordinates():
    grid = build_patch_grid((2, 3))

    assert grid.shape == (6, 2)
    assert torch.allclose(grid[0], torch.tensor([0.0, 0.0]))
    assert torch.allclose(grid[-1], torch.tensor([1.0, 1.0]))


def test_keypoint_relative_position_bias_shapes_and_gradients():
    module = KeypointRelativePositionBias(
        num_heads=2,
        grid_size=(2, 2),
        num_keypoints=5,
        hidden_dim=8,
        include_cls_token=True,
    )
    keypoints = torch.rand(3, 5, 2)

    pairwise_bias = module(keypoints, pairwise=True)
    token_bias = module(keypoints, pairwise=False)
    loss = pairwise_bias.sum() + token_bias.sum()
    loss.backward()

    assert pairwise_bias.shape == (3, 2, 5, 5)
    assert token_bias.shape == (3, 2, 5)
    assert module.mlp[0].weight.grad is not None
