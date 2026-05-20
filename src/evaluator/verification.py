from pathlib import Path

import numpy as np
import pandas as pd

from .base import BaseEvaluator
from .ddp import DDPEvaluator
from .protocols import read_pair_protocol


def _config_value(config, key, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def normalize_embeddings(embeddings, eps=1e-12):
    embeddings = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, eps)


def cosine_pair_scores(embeddings, pairs, normalize=True):
    embeddings = np.asarray(embeddings, dtype=np.float32)
    pairs = np.asarray(pairs, dtype=np.int64)
    if normalize:
        embeddings = normalize_embeddings(embeddings)
    return np.sum(embeddings[pairs[:, 0]] * embeddings[pairs[:, 1]], axis=1)


def calculate_accuracy(scores, is_same, threshold):
    scores = np.asarray(scores)
    is_same = np.asarray(is_same, dtype=bool)
    predictions = scores >= threshold
    return float(np.mean(predictions == is_same))


def best_accuracy_threshold(scores, is_same):
    scores = np.asarray(scores, dtype=np.float64)
    is_same = np.asarray(is_same, dtype=bool)
    if scores.size == 0:
        raise ValueError("Verification requires at least one pair")

    eps = np.finfo(scores.dtype).eps
    thresholds = np.unique(scores)
    thresholds = np.concatenate(
        ([scores.max() + eps], thresholds[::-1], [scores.min() - eps])
    )
    accuracies = np.array(
        [calculate_accuracy(scores, is_same, threshold) for threshold in thresholds]
    )
    best_idx = int(np.argmax(accuracies))
    return float(accuracies[best_idx]), float(thresholds[best_idx])


def cross_validation_accuracy(scores, is_same, n_folds=10):
    scores = np.asarray(scores, dtype=np.float64)
    is_same = np.asarray(is_same, dtype=bool)
    if scores.size < 2 or n_folds <= 1:
        accuracy, threshold = best_accuracy_threshold(scores, is_same)
        return accuracy, threshold, 0.0

    n_folds = min(int(n_folds), scores.size)
    indices = np.arange(scores.size)
    fold_indices = np.array_split(indices, n_folds)
    accuracies = []
    thresholds = []
    for valid_idx in fold_indices:
        train_idx = np.setdiff1d(indices, valid_idx, assume_unique=True)
        if train_idx.size == 0:
            accuracy, threshold = best_accuracy_threshold(scores, is_same)
        else:
            _, threshold = best_accuracy_threshold(
                scores[train_idx], is_same[train_idx]
            )
            accuracy = calculate_accuracy(
                scores[valid_idx],
                is_same[valid_idx],
                threshold,
            )
        accuracies.append(accuracy)
        thresholds.append(threshold)

    return (
        float(np.mean(accuracies)),
        float(np.mean(thresholds)),
        float(np.std(accuracies)),
    )


def tar_at_far(scores, is_same, fars=(1e-4, 1e-3, 1e-2)):
    scores = np.asarray(scores, dtype=np.float64)
    is_same = np.asarray(is_same, dtype=bool)
    positive_scores = scores[is_same]
    negative_scores = scores[~is_same]

    if positive_scores.size == 0 or negative_scores.size == 0:
        raise ValueError("TAR@FAR requires both positive and negative pairs")

    eps = np.finfo(scores.dtype).eps
    thresholds = np.unique(scores)
    thresholds = np.concatenate(
        ([scores.max() + eps], thresholds[::-1], [scores.min() - eps])
    )

    tars = np.array([np.mean(positive_scores >= threshold) for threshold in thresholds])
    observed_fars = np.array(
        [np.mean(negative_scores >= threshold) for threshold in thresholds]
    )

    metrics = {}
    for far in fars:
        eligible = np.flatnonzero(observed_fars <= far)
        if eligible.size == 0:
            best_idx = 0
        else:
            best_local_idx = int(np.argmax(tars[eligible]))
            best_idx = int(eligible[best_local_idx])
        metrics[float(far)] = {
            "tar": float(tars[best_idx]),
            "far": float(observed_fars[best_idx]),
            "threshold": float(thresholds[best_idx]),
        }

    return metrics


def evaluate_verification_pairs(
    embeddings,
    pairs,
    is_same,
    fars=(1e-4, 1e-3, 1e-2),
    normalize=True,
    n_folds=10,
):
    scores = cosine_pair_scores(embeddings, pairs, normalize=normalize)
    accuracy, threshold, accuracy_std = cross_validation_accuracy(
        scores,
        is_same,
        n_folds=n_folds,
    )
    far_metrics = tar_at_far(scores, is_same, fars=fars)

    metrics = {
        "accuracy": round(accuracy, 5),
        "accuracy_std": round(accuracy_std, 5),
        "accuracy_threshold": round(threshold, 5),
        "accuracy_folds": int(min(max(n_folds, 1), len(scores))),
        "n_pairs": int(len(scores)),
    }
    for far, values in far_metrics.items():
        far_key = f"{far:g}"
        metrics[f"TAR@FAR={far_key}"] = round(values["tar"], 5)
        metrics[f"FAR@FAR={far_key}"] = round(values["far"], 5)
        metrics[f"threshold@FAR={far_key}"] = round(values["threshold"], 5)

    return metrics


def _parse_pair_label(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer, float, np.floating)):
        return bool(int(value))

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "same", "positive"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "different", "negative"}:
        return False
    raise ValueError(f"Could not parse verification pair label: {value}")


def _build_file_name_index(file_names):
    exact = {}
    basename_to_indices = {}
    for index, file_name in enumerate(file_names):
        file_name = str(file_name)
        exact[file_name] = index
        basename = Path(file_name).name
        basename_to_indices.setdefault(basename, []).append(index)

    for basename, indices in basename_to_indices.items():
        if len(indices) == 1 and basename not in exact:
            exact[basename] = indices[0]
    return exact


def load_pairs_csv(
    pairs_path,
    file_names,
    file1_col="file1",
    file2_col="file2",
    label_col="is_same",
    protocol="csv",
    protocol_kwargs=None,
):
    pairs_path = Path(pairs_path)
    protocol_kwargs = {} if protocol_kwargs is None else protocol_kwargs
    pairs_df = read_pair_protocol(pairs_path, protocol=protocol, **protocol_kwargs)
    file_name_to_index = _build_file_name_index(file_names)

    pairs = []
    labels = []
    missing = []
    for row in pairs_df.itertuples(index=False):
        row_dict = row._asdict()
        file1 = str(row_dict[file1_col])
        file2 = str(row_dict[file2_col])
        if file1 not in file_name_to_index or file2 not in file_name_to_index:
            missing.append((file1, file2))
            continue

        pairs.append((file_name_to_index[file1], file_name_to_index[file2]))
        labels.append(_parse_pair_label(row_dict[label_col]))

    if missing:
        preview = ", ".join(f"{left}|{right}" for left, right in missing[:5])
        raise ValueError(
            f"{len(missing)} verification pairs reference missing files: {preview}"
        )
    if not pairs:
        raise ValueError(f"No verification pairs were loaded from {pairs_path}")

    return np.asarray(pairs, dtype=np.int64), np.asarray(labels, dtype=bool)


def build_label_pairs(labels, max_pairs=None):
    labels = np.asarray(labels)
    pairs = []
    is_same = []
    for left in range(len(labels)):
        for right in range(left + 1, len(labels)):
            pairs.append((left, right))
            is_same.append(labels[left] == labels[right])
            if max_pairs is not None and len(pairs) >= max_pairs:
                return np.asarray(pairs, dtype=np.int64), np.asarray(
                    is_same, dtype=bool
                )

    if not pairs:
        raise ValueError("At least two embeddings are required to build label pairs")
    return np.asarray(pairs, dtype=np.int64), np.asarray(is_same, dtype=bool)


class VerificationEvaluatorMixin:
    def _init_verification(self, config):
        evaluator_config = config.evaluation.evaluator
        self.save_results = _config_value(evaluator_config, "save_results", False)
        self.save_embeddings = _config_value(evaluator_config, "save_embeddings", False)
        self.fars = _config_value(evaluator_config, "fars", [1e-4, 1e-3, 1e-2])
        self.pairs_path = _config_value(evaluator_config, "pairs_path", None)
        self.pair_protocol = _config_value(evaluator_config, "pair_protocol", "csv")
        self.max_pairs = _config_value(evaluator_config, "max_pairs", None)
        self.normalize = _config_value(evaluator_config, "normalize", True)
        self.accuracy_folds = int(_config_value(evaluator_config, "accuracy_folds", 10))
        self.protocol_kwargs = dict(
            _config_value(evaluator_config, "protocol_kwargs", {}) or {}
        )
        pair_columns = _config_value(evaluator_config, "pair_columns", {})
        self.file1_col = _config_value(pair_columns, "file1", "file1")
        self.file2_col = _config_value(pair_columns, "file2", "file2")
        self.label_col = _config_value(pair_columns, "label", "is_same")

    def _get_pairs(self, labels, file_names):
        if self.pairs_path:
            return load_pairs_csv(
                self.pairs_path,
                file_names,
                file1_col=self.file1_col,
                file2_col=self.file2_col,
                label_col=self.label_col,
                protocol=self.pair_protocol,
                protocol_kwargs=self.protocol_kwargs,
            )
        return build_label_pairs(labels, max_pairs=self.max_pairs)

    def evaluate(self, data_info):
        embeddings, labels, file_names = self.generate_embeddings(data_info)
        if self.pca is not None:
            embeddings = self.pca.transform(embeddings)

        accelerator = getattr(self, "accelerator", None)
        if accelerator is not None and not accelerator.is_local_main_process:
            return {}

        pairs, is_same = self._get_pairs(labels, file_names)
        metrics = evaluate_verification_pairs(
            embeddings,
            pairs,
            is_same,
            fars=self.fars,
            normalize=self.normalize,
            n_folds=self.accuracy_folds,
        )

        if self.save_results:
            df_metrics = pd.DataFrame(metrics.items(), columns=["metric", "score"])
            df_metrics.to_csv(
                self.save_dir / f"{data_info.dataset_name}_verification_metrics.csv",
                index=False,
            )

        return metrics


class FaceVerificationEvaluator(VerificationEvaluatorMixin, BaseEvaluator):
    def __init__(
        self,
        config,
        model=None,
        save_dir="./",
        device="cpu",
        model_info=None,
        is_eval=True,
        pca=None,
    ):
        super().__init__(
            config=config,
            model=model,
            save_dir=save_dir,
            device=device,
            model_info=model_info,
            is_eval=False,
            pca=pca,
        )
        self._init_verification(config)


class DDPFaceVerificationEvaluator(VerificationEvaluatorMixin, DDPEvaluator):
    def __init__(
        self,
        config,
        model,
        save_dir="./",
        device="cpu",
        accelerator=None,
        is_eval=True,
        pca=None,
    ):
        super().__init__(
            config=config,
            model=model,
            save_dir=save_dir,
            device=device,
            accelerator=accelerator,
            is_eval=False,
            pca=pca,
        )
        self._init_verification(config)
