from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from data.face_benchmarks import (
    prepare_ijb,
    prepare_lfw_like,
    prepare_megaface,
    prepare_whitespace_verification,
)
from tools.benchmark_face_verification import run_face_verification_benchmark


def test_prepare_lfw_like_outputs_dataset_info_and_pairs(tmp_path):
    root_dir = tmp_path / "lfw"
    (root_dir / "Alice").mkdir(parents=True)
    (root_dir / "Bob").mkdir(parents=True)
    pairs_path = tmp_path / "pairs.txt"
    pairs_path.write_text("10 300\nAlice 1 2\nAlice 1 Bob 1\n")

    result = prepare_lfw_like(root_dir, pairs_path, tmp_path / "prepared")

    dataset_info = pd.read_csv(result["dataset_info"])
    pairs = pd.read_csv(result["pairs"])

    assert result["n_images"] == 3
    assert result["n_pairs"] == 2
    assert dataset_info["label"].tolist() == ["Alice", "Alice", "Bob"]
    assert pairs["is_same"].tolist() == [True, False]


def test_prepare_whitespace_verification_outputs_normalized_pairs(tmp_path):
    pairs_path = tmp_path / "cfp_pairs.txt"
    pairs_path.write_text(
        "frontal/Alice/001.jpg profile/Alice/001.jpg 1\n"
        "frontal/Alice/001.jpg profile/Bob/001.jpg 0\n"
    )

    result = prepare_whitespace_verification(
        tmp_path / "cfp",
        pairs_path,
        tmp_path / "prepared",
    )

    pairs = pd.read_csv(result["pairs"])

    assert result["n_images"] == 3
    assert pairs["file2"].tolist() == [
        "profile/Alice/001.jpg",
        "profile/Bob/001.jpg",
    ]


def test_prepare_ijb_outputs_csv_protocol_files(tmp_path):
    metadata_path = tmp_path / "ijb_face_tid_mid.txt"
    pairs_path = tmp_path / "ijb_template_pair_label.txt"
    metadata_path.write_text("a.jpg 1 10\nb.jpg 2 20\n")
    pairs_path.write_text("1 2 0\n")

    result = prepare_ijb(
        tmp_path / "ijb",
        metadata_path,
        pairs_path,
        tmp_path / "prepared",
    )

    metadata = pd.read_csv(result["metadata"])
    pairs = pd.read_csv(result["pairs"])

    assert result["n_templates"] == 2
    assert metadata["template_id"].tolist() == [1, 2]
    assert pairs["is_same"].tolist() == [0]


def test_prepare_megaface_scans_split_roots(tmp_path):
    probe_root = tmp_path / "probe"
    gallery_root = tmp_path / "gallery"
    (probe_root / "id1").mkdir(parents=True)
    (gallery_root / "id1").mkdir(parents=True)
    (probe_root / "id1" / "a.jpg").write_bytes(b"")
    (gallery_root / "id1" / "b.jpg").write_bytes(b"")

    manifest = prepare_megaface(
        tmp_path / "prepared",
        probe_root=probe_root,
        gallery_root=gallery_root,
    )

    assert manifest["probe"]["n_images"] == 1
    assert manifest["gallery"]["n_labels"] == 1


def test_lfw_preparation_feeds_offline_verification_benchmark(tmp_path):
    root_dir = tmp_path / "lfw"
    pairs_path = tmp_path / "pairs.txt"
    pairs_path.write_text("10 300\nAlice 1 2\nAlice 1 Bob 1\n")
    prepared = prepare_lfw_like(root_dir, pairs_path, tmp_path / "prepared")
    dataset_info = pd.read_csv(prepared["dataset_info"])
    embeddings_path = tmp_path / "embeddings.npz"
    embedding_map = {
        "Alice/Alice_0001.jpg": [1.0, 0.0],
        "Alice/Alice_0002.jpg": [0.95, 0.05],
        "Bob/Bob_0001.jpg": [0.0, 1.0],
    }
    embeddings = np.asarray(
        [embedding_map[file_name] for file_name in dataset_info["file_name"]],
        dtype=np.float32,
    )
    np.savez(
        embeddings_path,
        embeddings=embeddings,
        file_names=dataset_info["file_name"].to_numpy(),
    )

    metrics = run_face_verification_benchmark(
        embeddings_path,
        prepared["pairs"],
        fars=(0.0, 1.0),
    )

    assert metrics["accuracy"] == 1.0
