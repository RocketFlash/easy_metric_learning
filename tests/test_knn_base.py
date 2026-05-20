import importlib.util
from pathlib import Path

import numpy as np


def load_base_knn_module():
    module_path = (
        Path(__file__).resolve().parents[1] / "src" / "evaluator" / "knn" / "base.py"
    )
    spec = importlib.util.spec_from_file_location("base_knn_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_base_knn():
    return load_base_knn_module().BaseKNN


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


def test_cosine_similarity_chunks_sorts_asymmetric_gallery_results():
    module = load_base_knn_module()
    queries = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    gallery = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.8, 0.2],
        ],
        dtype=np.float32,
    )

    distances, indices = module.cosine_similarity_chunks(
        queries,
        gallery,
        n_chunks=1,
        top_n=2,
    )

    assert indices.tolist() == [[1, 2], [0, 2]]
    assert np.all(distances[:, 0] >= distances[:, 1])


def test_remove_self_neighbors_drops_actual_query_index_not_first_column():
    module = load_base_knn_module()
    indices = np.array([[1, 0], [1, 0]])
    distances = np.array([[0.9, 1.0], [1.0, 0.9]])

    filtered_indices, filtered_distances = module.remove_self_neighbors(
        indices,
        distances,
        top_k=1,
    )

    assert filtered_indices.tolist() == [[1], [0]]
    assert filtered_distances.tolist() == [[0.9], [0.9]]
