import pytest

np = pytest.importorskip("numpy")

from tools.benchmark_face_verification import run_face_verification_benchmark
from tools.benchmark_ijb import run_ijb_template_benchmark
from tools.benchmark_megaface import run_megaface_benchmark


def test_face_verification_benchmark_runs_from_npz_and_pairs(tmp_path):
    embeddings_path = tmp_path / "embeddings.npz"
    pairs_path = tmp_path / "pairs.csv"
    np.savez(
        embeddings_path,
        embeddings=np.asarray(
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
            dtype=np.float32,
        ),
        file_names=np.asarray(["a/1.jpg", "a/2.jpg", "b/1.jpg"]),
    )
    pairs_path.write_text(
        "file1,file2,is_same\n" "a/1.jpg,a/2.jpg,1\n" "a/1.jpg,b/1.jpg,0\n"
    )

    metrics = run_face_verification_benchmark(
        embeddings_path,
        pairs_path,
        fars=(0.0, 1.0),
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["n_pairs"] == 2


def test_ijb_benchmark_runs_from_official_txt_files(tmp_path):
    embeddings_path = tmp_path / "embeddings.npz"
    metadata_path = tmp_path / "ijb_face_tid_mid.txt"
    pairs_path = tmp_path / "ijb_template_pair_label.txt"
    np.savez(
        embeddings_path,
        embeddings=np.asarray(
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
            dtype=np.float32,
        ),
        file_names=np.asarray(["a.jpg", "b.jpg", "c.jpg"]),
    )
    metadata_path.write_text("a.jpg 1 10\nb.jpg 2 20\nc.jpg 3 30\n")
    pairs_path.write_text("1 2 1\n1 3 0\n")

    metrics = run_ijb_template_benchmark(
        embeddings_path,
        metadata_path,
        pairs_path,
        fars=(0.0, 1.0),
    )

    assert metrics["n_templates"] == 3
    assert metrics["n_pairs"] == 2


def test_megaface_benchmark_runs_from_npz_files(tmp_path):
    probe_path = tmp_path / "probe.npz"
    gallery_path = tmp_path / "gallery.npz"
    distractor_path = tmp_path / "distractor.npz"
    np.savez(
        probe_path,
        embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        labels=np.asarray(["id1", "unknown"]),
    )
    np.savez(
        gallery_path,
        embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
        labels=np.asarray(["id1"]),
    )
    np.savez(
        distractor_path,
        embeddings=np.asarray([[0.0, 1.0]], dtype=np.float32),
        labels=np.asarray(["d1"]),
    )

    metrics = run_megaface_benchmark(
        probe_path,
        gallery_path,
        distractor_path=distractor_path,
        ranks=(1,),
        fpirs=(1.0,),
    )

    assert metrics["CMC@1"] == 1.0
    assert "TPIR@FPIR=1" in metrics
