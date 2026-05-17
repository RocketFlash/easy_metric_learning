from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
cv2 = pytest.importorskip("cv2")
torch = pytest.importorskip("torch")
pytest.importorskip("hydra")
pytest.importorskip("omegaconf")
pytest.importorskip("albumentations")

import src.data as data_module
from src.data.dataset import base as base_dataset_module
from src.data.dataset.base import BaseDataset
from src.data.utils import collate_fn


class DummyDataset:
    label_ids = [0, 0, 1, 1]


def test_train_loader_disables_shuffle_when_sampler_is_configured(monkeypatch):
    captured = {}
    sampler = object()

    monkeypatch.setattr(
        data_module,
        "get_dataset",
        lambda root_dir, df_names, transform, labels_to_ids, dataset_config: DummyDataset(),
    )
    monkeypatch.setattr(
        data_module, "get_sampler", lambda labels, sampler_config: sampler
    )

    def fake_data_loader(**kwargs):
        captured.update(kwargs)
        return "loader"

    monkeypatch.setattr(data_module, "DataLoader", fake_data_loader)

    loader, dataset = data_module.get_loader(
        root_dir="/tmp/data",
        df_names=[],
        transform=None,
        dataset_config=SimpleNamespace(),
        dataloader_config=SimpleNamespace(
            sampler=SimpleNamespace(type="balanced"),
            batch_size=4,
            n_workers=0,
            pin_memory=False,
        ),
        split="train",
    )

    assert loader == "loader"
    assert isinstance(dataset, DummyDataset)
    assert captured["sampler"] is sampler
    assert captured["shuffle"] is False
    assert captured["drop_last"] is True


def test_base_dataset_reads_grayscale_images_as_rgb(tmp_path):
    image = np.arange(20, dtype=np.uint8).reshape(4, 5)
    cv2.imwrite(str(tmp_path / "gray.png"), image)
    df = pd.DataFrame({"file_name": ["gray.png"], "label": ["class_a"]})
    dataset = BaseDataset(tmp_path, df)

    sample, target, file_name = dataset[0]

    assert sample.shape == (4, 5, 3)
    assert np.array_equal(sample[:, :, 0], image)
    assert np.array_equal(sample[:, :, 1], image)
    assert np.array_equal(sample[:, :, 2], image)
    assert target.item() == 0
    assert file_name == "gray.png"


def test_base_dataset_reads_gif_images_as_uint8_rgb(monkeypatch, tmp_path):
    gif_data = np.array(
        [
            [[0.0, 0.5, 1.0, 1.0]],
            [[1.0, 0.0, 0.5, 1.0]],
        ],
        dtype=np.float32,
    )
    monkeypatch.setattr(base_dataset_module.plt, "imread", lambda image_path: gif_data)
    df = pd.DataFrame({"file_name": ["sample.gif"], "label": ["class_a"]})
    dataset = BaseDataset(tmp_path, df)

    sample, _, _ = dataset[0]

    assert sample.dtype == np.uint8
    assert sample.shape == (2, 1, 3)
    assert sample.tolist() == [[[0, 128, 255]], [[255, 0, 128]]]


def test_base_dataset_warns_and_returns_none_for_corrupt_images(tmp_path):
    df = pd.DataFrame({"file_name": ["missing.png"], "label": ["class_a"]})
    dataset = BaseDataset(tmp_path, df)

    with pytest.warns(RuntimeWarning, match="Corrupted image"):
        assert dataset[0] is None


def test_collate_fn_warns_when_dropping_bad_samples():
    valid_sample = (
        torch.zeros(2, 2, 3),
        torch.tensor(0, dtype=torch.long),
        "valid.png",
    )

    with pytest.warns(RuntimeWarning, match="Dropped 1 invalid samples"):
        images, labels, file_names = collate_fn([None, valid_sample])

    assert images.shape == (1, 2, 2, 3)
    assert labels.tolist() == [0]
    assert file_names == ("valid.png",)


def test_collate_fn_raises_when_all_samples_are_invalid():
    with pytest.raises(ValueError, match="All samples in batch failed"):
        collate_fn([None, None])
