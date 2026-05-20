import numpy as np


def _normalize(embeddings, eps=1e-12):
    embeddings = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, eps)


def evaluate_identification(
    probe_embeddings,
    probe_labels,
    gallery_embeddings,
    gallery_labels,
    ranks=(1, 10),
    fpirs=(1e-3, 1e-2),
    normalize=True,
):
    probe_embeddings = np.asarray(probe_embeddings, dtype=np.float32)
    gallery_embeddings = np.asarray(gallery_embeddings, dtype=np.float32)
    probe_labels = np.asarray(probe_labels)
    gallery_labels = np.asarray(gallery_labels)

    if normalize:
        probe_embeddings = _normalize(probe_embeddings)
        gallery_embeddings = _normalize(gallery_embeddings)

    similarities = probe_embeddings @ gallery_embeddings.T
    max_rank = min(max(int(rank) for rank in ranks), gallery_embeddings.shape[0])
    top_indices = np.argpartition(-similarities, kth=max_rank - 1, axis=1)[:, :max_rank]
    top_values = np.take_along_axis(similarities, top_indices, axis=1)
    top_order = np.argsort(-top_values, axis=1)
    ranked_indices = np.take_along_axis(top_indices, top_order, axis=1)
    ranked_labels = gallery_labels[ranked_indices]
    gallery_label_set = set(gallery_labels.tolist())
    genuine_mask = np.asarray(
        [label in gallery_label_set for label in probe_labels.tolist()]
    )
    imposter_mask = ~genuine_mask

    metrics = {}
    for rank in ranks:
        rank = min(int(rank), max_rank)
        correct = ranked_labels[:, :rank] == probe_labels[:, None]
        if genuine_mask.any():
            metrics[f"CMC@{rank}"] = float(correct[genuine_mask].any(axis=1).mean())
        else:
            metrics[f"CMC@{rank}"] = 0.0

    top_scores = similarities.max(axis=1)
    top_labels = ranked_labels[:, 0]
    imposter_scores = top_scores[imposter_mask]
    for fpir in fpirs:
        if imposter_scores.size == 0:
            threshold = float("inf")
            observed_fpir = 0.0
        else:
            threshold = float(np.quantile(imposter_scores, 1.0 - fpir))
            observed_fpir = float(np.mean(imposter_scores >= threshold))

        if genuine_mask.any():
            true_identify = (top_scores[genuine_mask] >= threshold) & (
                top_labels[genuine_mask] == probe_labels[genuine_mask]
            )
            tpir = float(true_identify.mean())
        else:
            tpir = 0.0

        key = f"{float(fpir):g}"
        metrics[f"TPIR@FPIR={key}"] = tpir
        metrics[f"FPIR@FPIR={key}"] = observed_fpir
        metrics[f"threshold@FPIR={key}"] = threshold

    return metrics
