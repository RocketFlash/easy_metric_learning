import pytest

np = pytest.importorskip("numpy")

from src.evaluator.identification import evaluate_identification


def test_identification_metrics_report_cmc_and_tpir():
    gallery_embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )
    gallery_labels = np.array(["a", "b", "distractor"])
    probe_embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.7, 0.7],
        ],
        dtype=np.float32,
    )
    probe_labels = np.array(["a", "b", "unknown", "unknown"])

    metrics = evaluate_identification(
        probe_embeddings,
        probe_labels,
        gallery_embeddings,
        gallery_labels,
        ranks=(1, 2),
        fpirs=(0.5,),
    )

    assert metrics["CMC@1"] == 1.0
    assert metrics["CMC@2"] == 1.0
    assert "TPIR@FPIR=0.5" in metrics
    assert "threshold@FPIR=0.5" in metrics
