import pytest

torch = pytest.importorskip("torch")

from src.trainer.targets import get_keypoints, move_to_device, split_targets


def test_split_targets_supports_tensor_labels():
    labels = torch.tensor([0, 1])

    split_labels, extras = split_targets(labels)

    assert split_labels is labels
    assert extras == {}


def test_split_targets_supports_keypoint_metadata():
    keypoints = torch.rand(2, 5, 2)
    targets = {"label": torch.tensor([0, 1]), "keypoints": keypoints}

    labels, extras = split_targets(targets)
    extras = move_to_device(extras, torch.device("cpu"))

    assert labels.tolist() == [0, 1]
    assert get_keypoints(extras).shape == (2, 5, 2)
