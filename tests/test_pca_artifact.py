import pytest

np = pytest.importorskip("numpy")

from src.pca import PCATransformer, load_pca, save_pca


class DummyPCA:
    components_ = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    mean_ = np.array([1.0, 2.0], dtype=np.float32)
    explained_variance_ = np.array([1.0, 1.0], dtype=np.float32)
    singular_values_ = np.array([1.0, 1.0], dtype=np.float32)
    n_components_ = 2
    n_features_in_ = 2
    n_samples_ = 2
    whiten = False


class DummyWhitenPCA(DummyPCA):
    explained_variance_ = np.array([4.0, 1.0], dtype=np.float32)
    whiten = True


def test_pca_artifact_round_trips_without_pickle(tmp_path):
    pca_path = tmp_path / "pca.npz"

    save_pca(DummyPCA(), pca_path)
    pca = load_pca(pca_path)

    assert isinstance(pca, PCATransformer)
    transformed = pca.transform(np.array([[2.0, 4.0]], dtype=np.float32))
    assert np.allclose(transformed, np.array([[1.0, 2.0]], dtype=np.float32))


def test_pca_loader_rejects_pickle_artifacts(tmp_path):
    pca_path = tmp_path / "pca.pkl"
    pca_path.write_bytes(b"not loaded")

    with pytest.raises(ValueError, match="Refusing to load pickle"):
        load_pca(pca_path)


def test_pca_artifact_round_trips_whitening(tmp_path):
    pca_path = tmp_path / "pca_whiten.npz"

    save_pca(DummyWhitenPCA(), pca_path)
    pca = load_pca(pca_path)

    transformed = pca.transform(np.array([[3.0, 5.0]], dtype=np.float32))
    assert np.allclose(transformed, np.array([[1.0, 3.0]], dtype=np.float32))


def test_pca_whitening_rejects_zero_explained_variance():
    pca = PCATransformer(
        components_=np.array([[1.0, 0.0]], dtype=np.float32),
        mean_=np.array([0.0, 0.0], dtype=np.float32),
        explained_variance_=np.array([0.0], dtype=np.float32),
        whiten=True,
    )

    with pytest.raises(ValueError, match="positive finite explained variance"):
        pca.transform(np.array([[1.0, 1.0]], dtype=np.float32))


def test_pca_artifact_matches_sklearn_fit_transform(tmp_path):
    decomposition = pytest.importorskip("sklearn.decomposition")
    data = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 2.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.float32,
    )
    sklearn_pca = decomposition.PCA(n_components=2, whiten=True).fit(data)
    pca_path = tmp_path / "sklearn_pca.npz"

    save_pca(sklearn_pca, pca_path)
    pca = load_pca(pca_path)

    assert np.allclose(
        pca.transform(data), sklearn_pca.transform(data).astype(np.float32), atol=1e-6
    )
