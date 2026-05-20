from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
torch = pytest.importorskip("torch")

from src.evaluator import get_evaluator
from src.evaluator.ijb import IJBTemplateEvaluator, aggregate_templates


class IdentityEmbeddingModel(torch.nn.Module):
    def get_embeddings(self, images):
        return images.float()


def make_config(metadata_path, pairs_path):
    return SimpleNamespace(
        debug=False,
        embeddings_size=2,
        evaluation=SimpleNamespace(
            evaluator=SimpleNamespace(
                type="ijb_template",
                save_results=False,
                save_embeddings=False,
                fars=[0.0],
                metadata_path=str(metadata_path),
                pairs_path=str(pairs_path),
                columns=SimpleNamespace(
                    file="file_name",
                    template="template_id",
                    media="media_id",
                    left="template1",
                    right="template2",
                    label="is_same",
                ),
            )
        ),
    )


def test_aggregate_templates_averages_media_before_template():
    embeddings = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], dtype=np.float32)
    template_ids = np.array([1, 1, 2])
    media_ids = np.array([10, 10, 20])

    output_ids, template_embeddings = aggregate_templates(
        embeddings, template_ids, media_ids=media_ids, normalize=False
    )

    assert output_ids == [1, 2]
    assert template_embeddings.tolist() == [[0.75, 0.25], [0.0, 1.0]]


def test_aggregate_templates_normalizes_frames_before_averaging():
    embeddings = np.array([[10.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    template_ids = np.array([1, 1])
    media_ids = np.array([10, 11])

    _, template_embeddings = aggregate_templates(
        embeddings, template_ids, media_ids=media_ids, normalize=True
    )

    expected = np.array([[1.0, 1.0]], dtype=np.float32)
    expected = expected / np.linalg.norm(expected, axis=1, keepdims=True)
    assert np.allclose(template_embeddings, expected)


def test_ijb_template_evaluator_runs_pair_protocol(tmp_path):
    metadata_path = tmp_path / "metadata.csv"
    pairs_path = tmp_path / "pairs.csv"
    pd.DataFrame(
        [
            {"file_name": "a1.jpg", "template_id": 1, "media_id": 1},
            {"file_name": "a2.jpg", "template_id": 1, "media_id": 2},
            {"file_name": "b1.jpg", "template_id": 2, "media_id": 3},
            {"file_name": "b2.jpg", "template_id": 2, "media_id": 4},
            {"file_name": "a3.jpg", "template_id": 3, "media_id": 5},
        ]
    ).to_csv(metadata_path, index=False)
    pd.DataFrame(
        [
            {"template1": 1, "template2": 3, "is_same": True},
            {"template1": 1, "template2": 2, "is_same": False},
        ]
    ).to_csv(pairs_path, index=False)

    data_info = SimpleNamespace(
        dataset_name="ijb_tiny",
        dataset_stats=SimpleNamespace(n_samples=5),
        ids_to_labels={0: "a", 1: "a", 2: "b", 3: "b", 4: "a"},
        dataloader=[
            (
                torch.tensor(
                    [
                        [1.0, 0.0],
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [0.0, 1.0],
                        [1.0, 0.0],
                    ]
                ),
                torch.tensor([0, 1, 2, 3, 4]),
                ["a1.jpg", "a2.jpg", "b1.jpg", "b2.jpg", "a3.jpg"],
            )
        ],
    )

    evaluator = get_evaluator(
        make_config(metadata_path, pairs_path),
        model=IdentityEmbeddingModel(),
        save_dir=tmp_path,
        device="cpu",
    )
    metrics = evaluator.evaluate(data_info)

    assert isinstance(evaluator, IJBTemplateEvaluator)
    assert metrics["accuracy"] == 1.0
    assert metrics["TAR@FAR=0"] == 1.0
    assert metrics["n_templates"] == 3
