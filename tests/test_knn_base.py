import importlib.util
from pathlib import Path

import numpy as np


def load_base_knn():
    module_path = (
        Path(__file__).resolve().parents[1] / "src" / "evaluator" / "knn" / "base.py"
    )
    spec = importlib.util.spec_from_file_location("base_knn_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BaseKNN


def test_base_knn_search_excludes_query_sample_and_uses_chunks(tmp_path):
    BaseKNN = load_base_knn()

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
    file_names = np.array(["a1", "a2", "b1", "b2"], dtype=object)

    knn = BaseKNN(K=[1], n_chunks=10, save_results=False, save_dir=tmp_path)
    result = knn.nearest_search(embeddings, labels, file_names=file_names)

    assert result["gt"].tolist() == ["a", "a", "b", "b"]
    assert [pred[0] for pred in result["prediction"]] == ["a", "a", "b", "b"]
    assert result["file_name"].tolist() == ["a1", "a2", "b1", "b2"]
