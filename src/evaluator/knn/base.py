import numpy as np
from tqdm.auto import tqdm
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def cosine_similarity_chunks(X, Y, n_chunks=5, top_n=5, sparse=False):
    ch_sz = X.shape[0]//n_chunks

    best_top_n_vals = None
    best_top_n_idxs = None

    for i in tqdm(range(n_chunks)):
        chunk = X[i*ch_sz:,:] if i==n_chunks-1 else X[i*ch_sz:(i+1)*ch_sz,:]
        cosine_sim_matrix_i = cosine_similarity(chunk, Y)
        best_top_n_vals, best_top_n_idxs = calculate_top_n(
            cosine_sim_matrix_i,
            best_top_n_vals,
            best_top_n_idxs,
            curr_zero_idx=(i*ch_sz),
            n=top_n
        )
    return best_top_n_vals, best_top_n_idxs


def calculate_top_n(sim_matrix,best_top_n_vals,
                               best_top_n_idxs,
                               curr_zero_idx=0,
                               n=10):
    n_rows, n_cols = sim_matrix.shape
    total_matrix_vals = sim_matrix
    total_matrix_idxs = np.tile(np.arange(n_rows).reshape(n_rows,1), (1,n_cols)).astype(int) + curr_zero_idx
    if curr_zero_idx>0:
        total_matrix_vals = np.vstack((total_matrix_vals, best_top_n_vals))
        total_matrix_idxs = np.vstack((total_matrix_idxs, best_top_n_idxs))
    res = np.argpartition(total_matrix_vals, -n, axis=0)[-n:]
    res_vals = np.take_along_axis(total_matrix_vals, res, axis=0)
    res_idxs = np.take_along_axis(total_matrix_idxs, res, axis=0)

    del res, total_matrix_idxs, total_matrix_vals
    return res_vals, res_idxs


class BaseKNN():
    def __init__(
            self,
            K=1,
            save_results=False,
            save_dir='./'
        ):
        if isinstance(K, int):
            K = [K]

        self.K = K
        self.save_results = save_results
        self.save_dir = save_dir


    def get_nearest_info(
            self, 
            labels,
            best_top_n_idxs,
            distances,
            file_names=None
        ):
        all_pred = []
        all_gts  = []
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
                columns=['file_name', 'gt', 'prediction', 'similarity']
            )
        else:
            df_nearest = pd.DataFrame(
                list(zip(all_gts, all_pred, all_dist)), 
                columns=['gt', 'prediction', 'similarity']
            )

        return df_nearest


    def nearest_search(
            self, 
            embeddings, 
            labels, 
            file_names=None, 
            dataset_name='dataset'
        ):
        
        top_k = max(self.K)
        
        best_top_n_vals, best_top_n_idxs = cosine_similarity_chunks(
            embeddings, 
            embeddings, 
            n_chunks=self.config.evaluation.n_chunks, 
            top_n=top_k+1
        )

        best_top_n_idxs = best_top_n_idxs.T
        distances = best_top_n_vals.T

        best_top_n_idxs = best_top_n_idxs[:, 1:]
        distances = distances[:, 1:]

        df_nearest = self.get_nearest_info(
            labels,
            best_top_n_idxs,
            distances,
            file_names=file_names
        )

        if self.save_results:
            df_nearest.to_feather(
                self.save_dir / f'{dataset_name}_top{top_k}.feather'
            )

        return df_nearest