import numpy as np

from .identification import evaluate_identification


def load_embedding_npz(path):
    data = np.load(path, allow_pickle=True)
    if "embeddings" not in data:
        raise ValueError(f"{path} must contain an embeddings array")
    labels = data["labels"] if "labels" in data else np.arange(len(data["embeddings"]))
    file_names = data["file_names"] if "file_names" in data else None
    return data["embeddings"], labels, file_names


def evaluate_megaface_npz(
    probe_path,
    gallery_path,
    distractor_path=None,
    ranks=(1, 10),
    fpirs=(1e-3, 1e-2),
):
    probe_embeddings, probe_labels, _ = load_embedding_npz(probe_path)
    gallery_embeddings, gallery_labels, _ = load_embedding_npz(gallery_path)

    if distractor_path is not None:
        distractor_embeddings, distractor_labels, _ = load_embedding_npz(
            distractor_path
        )
        distractor_labels = np.asarray(
            [f"distractor::{label}" for label in distractor_labels.tolist()]
        )
        gallery_embeddings = np.concatenate(
            [gallery_embeddings, distractor_embeddings], axis=0
        )
        gallery_labels = np.concatenate([gallery_labels, distractor_labels], axis=0)

    return evaluate_identification(
        probe_embeddings,
        probe_labels,
        gallery_embeddings,
        gallery_labels,
        ranks=ranks,
        fpirs=fpirs,
    )
