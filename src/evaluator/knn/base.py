import numpy as np
from tqdm.auto import tqdm
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from src.evaluator.rerank import is_rerank_enabled, rerank_from_config


def cosine_similarity_chunks(X, Y, n_chunks=5, top_n=5, sparse=False):
    n_chunks = max(1, min(int(n_chunks), X.shape[0]))
    top_n = min(top_n, Y.shape[0])
    ch_sz = X.shape[0] // n_chunks

    top_n_vals = []
    top_n_idxs = []

    for i in tqdm(range(n_chunks)):
        chunk = (
            X[i * ch_sz :, :]
            if i == n_chunks - 1
            else X[i * ch_sz : (i + 1) * ch_sz, :]
        )
        cosine_sim_matrix_i = cosine_similarity(chunk, Y)
        chunk_top_n_vals, chunk_top_n_idxs = calculate_top_n(
            cosine_sim_matrix_i,
            n=top_n,
        )
        top_n_vals.append(chunk_top_n_vals)
        top_n_idxs.append(chunk_top_n_idxs)
    return np.vstack(top_n_vals), np.vstack(top_n_idxs)


def calculate_top_n(
    sim_matrix, best_top_n_vals=None, best_top_n_idxs=None, curr_zero_idx=0, n=10
):
    del best_top_n_vals, best_top_n_idxs, curr_zero_idx
    n = min(n, sim_matrix.shape[1])
    top_n_idxs = np.argpartition(sim_matrix, -n, axis=1)[:, -n:]
    top_n_vals = np.take_along_axis(sim_matrix, top_n_idxs, axis=1)
    order = np.argsort(top_n_vals, axis=1)[:, ::-1]
    top_n_vals = np.take_along_axis(top_n_vals, order, axis=1)
    top_n_idxs = np.take_along_axis(top_n_idxs, order, axis=1)
    return top_n_vals, top_n_idxs


def remove_self_neighbors(best_top_n_idxs, distances, top_k):
    filtered_idxs = []
    filtered_distances = []
    for query_index, (neighbor_idxs, neighbor_distances) in enumerate(
        zip(best_top_n_idxs, distances)
    ):
        keep = neighbor_idxs != query_index
        filtered_idxs.append(neighbor_idxs[keep][:top_k])
        filtered_distances.append(neighbor_distances[keep][:top_k])
    return (
        np.asarray(filtered_idxs, dtype=best_top_n_idxs.dtype),
        np.asarray(filtered_distances, dtype=distances.dtype),
    )


class BaseKNN:
    def __init__(
        self,
        K=1,
        n_chunks=5,
        save_results=False,
        save_dir="./",
        rerank_config=None,
    ):
        if isinstance(K, int):
            K = [K]

        self.K = K
        self.n_chunks = n_chunks
        self.save_results = save_results
        self.save_dir = Path(save_dir)
        self.rerank_config = rerank_config

    def get_nearest_info(self, labels, best_top_n_idxs, distances, file_names=None):
        all_pred = []
        all_gts = []
        all_dist = []
        all_fnms = []

        for i in range(len(best_top_n_idxs)):
            all_pred.append(labels[best_top_n_idxs[i]])
            all_gts.append(labels[i])
            all_dist.append(distances[i])
            if file_names is not None:
                all_fnms.append(file_names[i])

        if file_names is not None:
            df_nearest = pd.DataFrame(
                list(zip(all_fnms, all_gts, all_pred, all_dist)),
                columns=["file_name", "gt", "prediction", "similarity"],
            )
        else:
            df_nearest = pd.DataFrame(
                list(zip(all_gts, all_pred, all_dist)),
                columns=["gt", "prediction", "similarity"],
            )

        return df_nearest

    def nearest_search(
        self, embeddings, labels, file_names=None, dataset_name="dataset"
    ):

        top_k = max(self.K)

        if is_rerank_enabled(self.rerank_config):
            best_top_n_idxs, distances = rerank_from_config(
                embeddings, top_k=top_k, config=self.rerank_config
            )
        else:
            best_top_n_vals, best_top_n_idxs = cosine_similarity_chunks(
                embeddings, embeddings, n_chunks=self.n_chunks, top_n=top_k + 1
            )

            best_top_n_idxs, distances = remove_self_neighbors(
                best_top_n_idxs, best_top_n_vals, top_k
            )

        df_nearest = self.get_nearest_info(
            labels, best_top_n_idxs, distances, file_names=file_names
        )

        if self.save_results:
            df_nearest.to_feather(self.save_dir / f"{dataset_name}_top{top_k}.feather")

        return df_nearest
