import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("faiss")

from src.evaluator.knn.faiss import FAISSKNN


@pytest.mark.parametrize("index_type", ["flat", "ivf", "hnsw"])
def test_faiss_knn_supports_index_types(index_type, tmp_path):
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

    knn = FAISSKNN(
        K=[1],
        n_workers=1,
        save_results=False,
        save_dir=tmp_path,
        index_type=index_type,
        nlist=2,
        nprobe=2,
        hnsw_m=8,
    )
    result = knn.nearest_search(embeddings, labels)

    assert result["gt"].tolist() == ["a", "a", "b", "b"]
    assert [prediction[0] for prediction in result["prediction"]] == [
        "a",
        "a",
        "b",
        "b",
    ]
