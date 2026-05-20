from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from src.sampler.advanced import HardNegativeSampler
from src.config import ConfigValidationError, validate_training_config
from src.trainer.hard_negative_cache import (
    maybe_refresh_hard_negative_cache,
    refresh_hard_negative_cache,
    should_refresh_hard_negative_cache,
)


class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.images = torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
                [0.85, 0.15],
                [0.8, 0.2],
            ],
            dtype=torch.float32,
        )
        self.labels = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
        self.file_names = [f"{index}.jpg" for index in range(len(self.labels))]

    def __getitem__(self, index):
        return self.images[index], self.labels[index], self.file_names[index]

    def __len__(self):
        return len(self.labels)


class IdentityEmbeddingModel(torch.nn.Module):
    def get_embeddings(self, images):
        return images


def make_cache_config(enabled=True):
    return SimpleNamespace(
        train=SimpleNamespace(
            trainer=SimpleNamespace(
                hard_negative_cache=SimpleNamespace(
                    enabled=enabled,
                    refresh_every=2,
                    start_epoch=2,
                    save_path="cache/hard_negative.npz",
                    max_batches=None,
                    update_sampler=True,
                )
            )
        )
    )


def test_hard_negative_sampler_refreshes_from_new_cache():
    labels = [0, 0, 1, 1, 2, 2]
    initial_embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.85, 0.15],
            [0.8, 0.2],
            [0.0, 1.0],
            [0.1, 0.9],
        ]
    )
    sampler = HardNegativeSampler(
        labels,
        embeddings=initial_embeddings,
        p=2,
        k=1,
        hard_negative_rate=1.0,
        length_before_new_iter=2,
    )

    assert sampler.hard_labels[0] == 1

    refreshed_embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
            [0.85, 0.15],
            [0.8, 0.2],
        ]
    )
    sampler.refresh(refreshed_embeddings, labels=labels)

    assert sampler.hard_labels[0] == 2


def test_refresh_hard_negative_cache_saves_npz_and_updates_sampler(tmp_path):
    dataset = EmbeddingDataset()
    sampler = HardNegativeSampler(
        labels=dataset.labels.numpy(),
        embeddings=np.asarray(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.85, 0.15],
                [0.8, 0.2],
                [0.0, 1.0],
                [0.1, 0.9],
            ],
            dtype=np.float32,
        ),
        p=2,
        k=1,
        hard_negative_rate=1.0,
        length_before_new_iter=2,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, sampler=sampler)
    cache_path = tmp_path / "cache.npz"

    result = refresh_hard_negative_cache(
        IdentityEmbeddingModel(),
        loader,
        cache_path,
        device="cpu",
    )

    cache = np.load(cache_path, allow_pickle=True)
    assert result["n_samples"] == len(dataset)
    assert result["sampler_refreshed"] is True
    assert cache["embeddings"].shape == (len(dataset), 2)
    assert sampler.hard_labels[0] == 2


def test_maybe_refresh_hard_negative_cache_respects_schedule(tmp_path):
    config = make_cache_config(enabled=True)
    dataset = EmbeddingDataset()
    loader = torch.utils.data.DataLoader(dataset, batch_size=3)

    assert should_refresh_hard_negative_cache(config, epoch=1) is False
    assert should_refresh_hard_negative_cache(config, epoch=2) is True
    assert should_refresh_hard_negative_cache(config, epoch=3) is False

    with pytest.warns(RuntimeWarning, match="no refreshable sampler"):
        result = maybe_refresh_hard_negative_cache(
            config,
            model=IdentityEmbeddingModel(),
            data_loader=loader,
            work_dir=tmp_path,
            device="cpu",
            epoch=2,
        )

    assert result["path"] == tmp_path / "cache/hard_negative.npz"
    assert result["path"].is_file()


def test_config_validation_rejects_invalid_hard_negative_cache_schedule():
    config = make_cache_config(enabled=True)
    config.train.trainer.hard_negative_cache.refresh_every = 0

    with pytest.raises(ConfigValidationError, match="refresh_every"):
        validate_training_config(config)


def test_config_validation_rejects_sampler_update_without_hard_negative_sampler():
    config = make_cache_config(enabled=True)
    config.train.trainer.hard_negative_cache.refresh_every = 1
    config.dataloader = SimpleNamespace(sampler=SimpleNamespace(type="pk"))

    with pytest.raises(ConfigValidationError, match="dataloader=hard_negative"):
        validate_training_config(config)
