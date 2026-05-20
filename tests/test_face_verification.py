from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.evaluator import get_evaluator
from src.evaluator.verification import (
    FaceVerificationEvaluator,
    build_label_pairs,
    cross_validation_accuracy,
    evaluate_verification_pairs,
    load_pairs_csv,
)


class IdentityEmbeddingModel(torch.nn.Module):
    def get_embeddings(self, images):
        return images.float()


def make_verification_config(pairs_path=None):
    return SimpleNamespace(
        debug=False,
        embeddings_size=2,
        evaluation=SimpleNamespace(
            evaluator=SimpleNamespace(
                type="face_verification",
                save_results=False,
                save_embeddings=False,
                fars=[0.0, 0.5],
                normalize=True,
                pairs_path=pairs_path,
                pair_columns=SimpleNamespace(
                    file1="file1",
                    file2="file2",
                    label="is_same",
                ),
                max_pairs=100,
            )
        ),
    )


def test_evaluate_verification_pairs_reports_tar_at_far():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    pairs = np.array([[0, 1], [2, 3], [0, 2], [1, 3]])
    is_same = np.array([True, True, False, False])

    metrics = evaluate_verification_pairs(embeddings, pairs, is_same, fars=[0.0, 0.5])

    assert metrics["accuracy"] == 1.0
    assert metrics["TAR@FAR=0"] == 1.0
    assert metrics["FAR@FAR=0"] == 0.0
    assert metrics["accuracy_folds"] == 4
    assert metrics["n_pairs"] == 4


def test_cross_validation_accuracy_selects_threshold_on_train_folds():
    scores = np.array([0.9, 0.8, 0.7, 0.1], dtype=np.float32)
    is_same = np.array([True, False, True, False])

    accuracy, threshold, accuracy_std = cross_validation_accuracy(
        scores,
        is_same,
        n_folds=2,
    )

    assert 0.0 <= accuracy <= 1.0
    assert 0.0 <= accuracy_std <= 0.5
    assert scores.min() <= threshold <= scores.max()


def test_load_pairs_csv_resolves_exact_and_basename_paths(tmp_path):
    pairs_path = tmp_path / "pairs.csv"
    pd.DataFrame(
        [
            {"file1": "a.jpg", "file2": "folder/b.jpg", "is_same": 1},
            {"file1": "c.jpg", "file2": "d.jpg", "is_same": "different"},
        ]
    ).to_csv(pairs_path, index=False)

    pairs, is_same = load_pairs_csv(
        pairs_path,
        file_names=np.array(["a.jpg", "folder/b.jpg", "root/c.jpg", "d.jpg"]),
    )

    assert pairs.tolist() == [[0, 1], [2, 3]]
    assert is_same.tolist() == [True, False]


def test_build_label_pairs_uses_labels_for_smoke_protocol():
    pairs, is_same = build_label_pairs(np.array(["a", "a", "b"]), max_pairs=3)

    assert pairs.tolist() == [[0, 1], [0, 2], [1, 2]]
    assert is_same.tolist() == [True, False, False]


def test_face_verification_evaluator_runs_pair_protocol(tmp_path):
    pairs_path = tmp_path / "pairs.csv"
    pd.DataFrame(
        [
            {"file1": "a1.jpg", "file2": "a2.jpg", "is_same": True},
            {"file1": "a1.jpg", "file2": "b1.jpg", "is_same": False},
        ]
    ).to_csv(pairs_path, index=False)

    data_info = SimpleNamespace(
        dataset_name="lfw_tiny",
        dataset_stats=SimpleNamespace(n_samples=4),
        ids_to_labels={0: "a", 1: "a", 2: "b", 3: "b"},
        dataloader=[
            (
                torch.tensor(
                    [
                        [1.0, 0.0],
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [0.0, 1.0],
                    ]
                ),
                torch.tensor([0, 1, 2, 3]),
                ["a1.jpg", "a2.jpg", "b1.jpg", "b2.jpg"],
            )
        ],
    )
    evaluator = get_evaluator(
        make_verification_config(pairs_path=str(pairs_path)),
        model=IdentityEmbeddingModel(),
        save_dir=tmp_path,
        device="cpu",
    )

    metrics = evaluator.evaluate(data_info)

    assert isinstance(evaluator, FaceVerificationEvaluator)
    assert metrics["accuracy"] == 1.0
    assert metrics["TAR@FAR=0"] == 1.0
