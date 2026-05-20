import numpy as np


def qmagface_pair_scores(embeddings, pairs, alpha=0.2, eps=1e-12):
    embeddings = np.asarray(embeddings, dtype=np.float32)
    pairs = np.asarray(pairs, dtype=np.int64)
    norms = np.linalg.norm(embeddings, axis=1)
    normalized = embeddings / np.maximum(norms[:, None], eps)
    cosine = np.sum(normalized[pairs[:, 0]] * normalized[pairs[:, 1]], axis=1)
    quality = np.minimum(norms[pairs[:, 0]], norms[pairs[:, 1]])
    quality = quality / max(float(norms.max()), eps)
    return cosine * (1.0 + alpha * quality)
