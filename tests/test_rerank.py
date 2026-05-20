from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from src.config import ConfigValidationError, validate_training_config
from src.evaluator.knn.base import BaseKNN
from src.evaluator.rerank import k_reciprocal_rerank


def test_k_reciprocal_rerank_returns_ranked_neighbors_without_self():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.95, 0.05],
            [0.0, 1.0],
            [0.05, 0.95],
        ],
        dtype=np.float32,
    )

    indices, similarities = k_reciprocal_rerank(
        embeddings,
        top_k=2,
        k1=2,
        k2=1,
        lambda_value=0.3,
    )

    assert indices.shape == (4, 2)
    assert similarities.shape == (4, 2)
    assert all(row_index not in row for row_index, row in enumerate(indices))
    assert indices[0, 0] == 1
    assert indices[2, 0] == 3


def test_base_knn_uses_rerank_config():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.95, 0.05],
            [0.0, 1.0],
            [0.05, 0.95],
        ],
        dtype=np.float32,
    )
    labels = np.array(["a", "a", "b", "b"], dtype=object)
    knn = BaseKNN(
        K=[1],
        save_results=False,
        rerank_config=SimpleNamespace(
            enabled=True,
            k1=2,
            k2=1,
            lambda_value=0.3,
        ),
    )

    result = knn.nearest_search(embeddings, labels)

    assert [prediction[0] for prediction in result["prediction"]] == [
        "a",
        "a",
        "b",
        "b",
    ]


def test_rerank_validation_rejects_invalid_lambda():
    config = SimpleNamespace(
        train=SimpleNamespace(trainer=SimpleNamespace()),
        evaluation=SimpleNamespace(
            knn=SimpleNamespace(
                rerank=SimpleNamespace(
                    enabled=True,
                    k1=20,
                    k2=6,
                    lambda_value=1.5,
                )
            )
        ),
    )

    with pytest.raises(ConfigValidationError, match="lambda_value"):
        validate_training_config(config)
