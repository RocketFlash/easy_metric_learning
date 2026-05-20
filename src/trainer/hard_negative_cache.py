from pathlib import Path
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..utils import is_main_process
from .targets import get_keypoints, move_to_device, split_targets


def _config_value(config, key, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def get_hard_negative_cache_config(config):
    trainer_config = _config_value(
        _config_value(config, "train", None), "trainer", None
    )
    return _config_value(trainer_config, "hard_negative_cache", None)


def is_hard_negative_cache_enabled(config):
    cache_config = get_hard_negative_cache_config(config)
    return bool(_config_value(cache_config, "enabled", False))


def should_refresh_hard_negative_cache(config, epoch):
    cache_config = get_hard_negative_cache_config(config)
    if not _config_value(cache_config, "enabled", False):
        return False
    start_epoch = int(_config_value(cache_config, "start_epoch", 1))
    refresh_every = int(_config_value(cache_config, "refresh_every", 1))
    if refresh_every <= 0:
        raise ValueError("hard_negative_cache.refresh_every must be positive")
    return epoch >= start_epoch and (epoch - start_epoch) % refresh_every == 0


def get_hard_negative_cache_path(config, work_dir):
    cache_config = get_hard_negative_cache_config(config)
    save_path = _config_value(cache_config, "save_path", None)
    if save_path is None:
        return Path(work_dir) / "hard_negative_cache.npz"
    save_path = Path(save_path)
    if not save_path.is_absolute():
        save_path = Path(work_dir) / save_path
    return save_path


def _cache_loader_from(data_loader):
    dataset = getattr(data_loader, "dataset", None)
    if dataset is None:
        return data_loader

    batch_size = getattr(data_loader, "batch_size", None)
    if batch_size is None:
        batch_sampler = getattr(data_loader, "batch_sampler", None)
        batch_size = getattr(batch_sampler, "batch_size", None)
    if batch_size is None:
        batch_size = 1

    num_workers = int(getattr(data_loader, "num_workers", 0))
    kwargs = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": False,
        "num_workers": num_workers,
        "collate_fn": getattr(data_loader, "collate_fn", None),
        "pin_memory": bool(getattr(data_loader, "pin_memory", False)),
        "drop_last": False,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(
            getattr(data_loader, "persistent_workers", False)
        )
        prefetch_factor = getattr(data_loader, "prefetch_factor", None)
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs)


def _find_refreshable_sampler(obj, seen=None):
    if obj is None:
        return None
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return None
    seen.add(obj_id)

    if hasattr(obj, "refresh"):
        return obj
    for attr in ("sampler", "batch_sampler"):
        child = getattr(obj, attr, None)
        sampler = _find_refreshable_sampler(child, seen=seen)
        if sampler is not None:
            return sampler
    return None


def _model_embeddings(model, images, target_extras):
    keypoints = get_keypoints(target_extras)
    if hasattr(model, "get_embeddings"):
        if keypoints is not None:
            return model.get_embeddings(images, keypoints=keypoints)
        return model.get_embeddings(images)

    output = model(images)
    if isinstance(output, (list, tuple)):
        output = output[-1]
    return output


def refresh_hard_negative_cache(
    model,
    data_loader,
    save_path,
    device="cpu",
    max_batches=None,
    accelerator=None,
    update_sampler=True,
):
    eval_model = accelerator.unwrap_model(model) if accelerator is not None else model
    cache_loader = _cache_loader_from(data_loader)
    was_training = eval_model.training
    eval_model.eval()

    embeddings = []
    labels = []
    file_names = []
    with torch.no_grad():
        for batch_index, (images, targets, fnames) in enumerate(cache_loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break

            targets, target_extras = split_targets(targets)
            images = images.to(device)
            targets = targets.to(device)
            target_extras = move_to_device(target_extras, device)

            batch_embeddings = _model_embeddings(eval_model, images, target_extras)
            if accelerator is not None:
                batch_embeddings = accelerator.gather_for_metrics(batch_embeddings)
                targets = accelerator.gather_for_metrics(targets)

            embeddings.append(batch_embeddings.detach().cpu().numpy())
            labels.append(targets.detach().cpu().numpy())
            file_names.extend([str(fname) for fname in fnames])

    if was_training:
        eval_model.train()

    if not embeddings:
        raise ValueError("No embeddings were generated for hard-negative cache")

    embeddings = np.concatenate(embeddings, axis=0).astype(np.float32)
    labels = np.concatenate(labels, axis=0)
    save_path = Path(save_path)
    if is_main_process(accelerator):
        save_path.parent.mkdir(exist_ok=True, parents=True)
        np.savez(
            save_path,
            embeddings=embeddings,
            labels=labels,
            file_names=np.asarray(file_names, dtype=object),
        )

    sampler = _find_refreshable_sampler(data_loader) if update_sampler else None
    if sampler is not None:
        sampler.refresh(embeddings=embeddings, labels=labels)
    elif update_sampler:
        warnings.warn(
            "hard_negative_cache.update_sampler=True, but the dataloader has no "
            "refreshable sampler; the cache was saved without updating sampling",
            RuntimeWarning,
        )

    return {
        "path": save_path,
        "n_samples": int(embeddings.shape[0]),
        "embedding_size": int(embeddings.shape[1]),
        "sampler_refreshed": sampler is not None,
    }


def maybe_refresh_hard_negative_cache(
    config,
    model,
    data_loader,
    work_dir,
    device,
    epoch,
    logger=None,
    accelerator=None,
):
    if not should_refresh_hard_negative_cache(config, epoch):
        return None
    cache_config = get_hard_negative_cache_config(config)
    save_path = get_hard_negative_cache_path(config, work_dir)
    result = refresh_hard_negative_cache(
        model=model,
        data_loader=data_loader,
        save_path=save_path,
        device=device,
        max_batches=_config_value(cache_config, "max_batches", None),
        accelerator=accelerator,
        update_sampler=_config_value(cache_config, "update_sampler", True),
    )
    if logger is not None and is_main_process(accelerator):
        logger.info(
            "Hard-negative cache refreshed: "
            f"{result['n_samples']} embeddings -> {result['path']}"
        )
    return result
