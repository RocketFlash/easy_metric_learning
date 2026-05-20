from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from src.config import ConfigValidationError, validate_training_config
from src.evaluator.ijb import load_template_metadata, load_template_pairs
from src.evaluator.megaface import evaluate_megaface_npz
from src.evaluator.protocols import read_lfw_style_pairs, read_whitespace_pairs
from src.utils import get_save_paths, load_ckp, save_ckp

torch = pytest.importorskip("torch")


def test_lfw_style_pairs_are_converted_to_file_pairs(tmp_path):
    pairs_path = tmp_path / "pairs.txt"
    pairs_path.write_text("10 300\nAlice 1 2\nAlice 1 Bob 1\n")

    pairs = read_lfw_style_pairs(pairs_path)

    assert pairs.to_dict("records") == [
        {
            "file1": "Alice/Alice_0001.jpg",
            "file2": "Alice/Alice_0002.jpg",
            "is_same": True,
        },
        {
            "file1": "Alice/Alice_0001.jpg",
            "file2": "Bob/Bob_0001.jpg",
            "is_same": False,
        },
    ]


def test_whitespace_pairs_support_cfp_style_paths(tmp_path):
    pairs_path = tmp_path / "cfp_pairs.txt"
    pairs_path.write_text(
        "frontal/a.jpg profile/a.jpg same\nfrontal/a.jpg profile/b.jpg 0\n"
    )

    pairs = read_whitespace_pairs(pairs_path)

    assert pairs["is_same"].tolist() == [True, False]
    assert pairs["file1"].tolist() == ["frontal/a.jpg", "frontal/a.jpg"]


def test_ijb_official_txt_metadata_and_pairs_load(tmp_path):
    metadata_path = tmp_path / "ijb_face_tid_mid.txt"
    pairs_path = tmp_path / "ijb_template_pair_label.txt"
    metadata_path.write_text("a.jpg 1 10\nb.jpg 2 20\n")
    pairs_path.write_text("1 2 0\n")

    template_ids, media_ids = load_template_metadata(
        metadata_path,
        file_names=["a.jpg", "b.jpg"],
    )
    pairs, is_same = load_template_pairs(
        pairs_path,
        {1: 0, 2: 1},
        "template1",
        "template2",
        "is_same",
    )

    assert template_ids.tolist() == [1, 2]
    assert media_ids.tolist() == [10, 20]
    assert pairs.tolist() == [[0, 1]]
    assert is_same.tolist() == [False]


def test_megaface_npz_protocol_includes_distractors(tmp_path):
    probe_path = tmp_path / "probe.npz"
    gallery_path = tmp_path / "gallery.npz"
    distractor_path = tmp_path / "distractor.npz"
    np.savez(
        probe_path,
        embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        labels=np.array(["a", "unknown"]),
    )
    np.savez(
        gallery_path,
        embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
        labels=np.array(["a"]),
    )
    np.savez(
        distractor_path,
        embeddings=np.array([[0.0, 1.0]], dtype=np.float32),
        labels=np.array(["d1"]),
    )

    metrics = evaluate_megaface_npz(
        probe_path,
        gallery_path,
        distractor_path=distractor_path,
        ranks=(1,),
        fpirs=(1.0,),
    )

    assert metrics["CMC@1"] == 1.0
    assert "TPIR@FPIR=1" in metrics


def test_config_validation_rejects_misconfigured_advanced_options():
    config = SimpleNamespace(
        ddp=False,
        train=SimpleNamespace(
            trainer=SimpleNamespace(
                fsdp=SimpleNamespace(enabled=True),
                compile=SimpleNamespace(enabled=False),
            )
        ),
        dataloader=SimpleNamespace(sampler=SimpleNamespace(type="default")),
        loss=SimpleNamespace(
            losses=[SimpleNamespace(name="multi_similarity", input="embeddings")]
        ),
        evaluation=SimpleNamespace(evaluator=SimpleNamespace(type="base")),
    )

    with pytest.raises(ConfigValidationError, match="fsdp"):
        validate_training_config(config)


def test_checkpoint_can_store_weight_only_averaged_artifact(tmp_path):
    model = torch.nn.Linear(2, 1)
    checkpoint_path = tmp_path / "averaged.pt"

    save_ckp(checkpoint_path, model=model, optimizer=None, epoch=2)
    restored_model = torch.nn.Linear(2, 1)
    _, _, epoch, _ = load_ckp(checkpoint_path, restored_model, device="cpu")

    assert epoch == 2
    assert (
        get_save_paths(tmp_path).last_averaged_weights_path.name == "last_averaged.pt"
    )
