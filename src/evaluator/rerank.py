import numpy as np


def _config_value(config, key, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def is_rerank_enabled(config):
    return bool(_config_value(config, "enabled", False))


def _normalize(embeddings, eps=1e-12):
    embeddings = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, eps)


def _reciprocal_neighbors(initial_rank, query_index, k):
    forward = initial_rank[query_index, : k + 1]
    reciprocal = [
        candidate
        for candidate in forward
        if query_index in initial_rank[candidate, : k + 1]
    ]
    return np.asarray(reciprocal, dtype=np.int32)


def k_reciprocal_rerank(
    embeddings,
    top_k,
    k1=20,
    k2=6,
    lambda_value=0.3,
):
    n_samples = embeddings.shape[0]
    top_k = min(int(top_k), max(0, n_samples - 1))
    if top_k == 0:
        return (
            np.empty((n_samples, 0), dtype=np.int64),
            np.empty((n_samples, 0), dtype=np.float32),
        )

    k1 = min(int(k1), max(1, n_samples - 1))
    k2 = min(int(k2), n_samples)
    lambda_value = float(lambda_value)

    embeddings = _normalize(embeddings)
    similarity = np.clip(embeddings @ embeddings.T, -1.0, 1.0)
    original_dist = np.maximum(0.0, 1.0 - similarity).astype(np.float32)
    initial_rank = np.argsort(original_dist, axis=1).astype(np.int32)

    weights = np.zeros_like(original_dist, dtype=np.float32)
    expansion_k = max(1, int(round(k1 / 2)))
    for query_index in range(n_samples):
        reciprocal = _reciprocal_neighbors(initial_rank, query_index, k1)
        reciprocal_expansion = reciprocal
        for candidate in reciprocal:
            candidate_reciprocal = _reciprocal_neighbors(
                initial_rank, int(candidate), expansion_k
            )
            if candidate_reciprocal.size == 0:
                continue
            overlap = np.intersect1d(candidate_reciprocal, reciprocal)
            if overlap.size > (2.0 / 3.0) * candidate_reciprocal.size:
                reciprocal_expansion = np.append(
                    reciprocal_expansion, candidate_reciprocal
                )
        reciprocal_expansion = np.unique(reciprocal_expansion)
        reciprocal_weights = np.exp(-original_dist[query_index, reciprocal_expansion])
        weight_sum = reciprocal_weights.sum()
        if weight_sum > 0:
            weights[query_index, reciprocal_expansion] = reciprocal_weights / weight_sum

    if k2 > 1:
        query_expanded = np.zeros_like(weights, dtype=np.float32)
        for query_index in range(n_samples):
            query_expanded[query_index] = weights[initial_rank[query_index, :k2]].mean(
                axis=0
            )
        weights = query_expanded

    inverted_index = [
        np.where(weights[:, candidate_index] > 0)[0]
        for candidate_index in range(n_samples)
    ]
    jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)
    for query_index in range(n_samples):
        temp_min = np.zeros(n_samples, dtype=np.float32)
        non_zero = np.where(weights[query_index] > 0)[0]
        for candidate_index in non_zero:
            related_queries = inverted_index[candidate_index]
            temp_min[related_queries] += np.minimum(
                weights[query_index, candidate_index],
                weights[related_queries, candidate_index],
            )
        jaccard_dist[query_index] = 1.0 - temp_min / np.maximum(2.0 - temp_min, 1e-12)

    final_dist = (1.0 - lambda_value) * jaccard_dist + lambda_value * original_dist
    np.fill_diagonal(final_dist, np.inf)
    nearest_indices = np.argsort(final_dist, axis=1)[:, :top_k]
    nearest_distances = np.take_along_axis(final_dist, nearest_indices, axis=1)
    similarities = 1.0 - nearest_distances
    return nearest_indices, similarities.astype(np.float32)


def rerank_from_config(embeddings, top_k, config):
    return k_reciprocal_rerank(
        embeddings,
        top_k=top_k,
        k1=_config_value(config, "k1", 20),
        k2=_config_value(config, "k2", 6),
        lambda_value=_config_value(config, "lambda_value", 0.3),
    )
