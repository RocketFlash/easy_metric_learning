import faiss
from .base import BaseKNN


class FAISSKNN(BaseKNN):
    def __init__(
            self, 
            K=1,
            n_workers=8,
            save_results=False,
            faiss_gpu=False,
            save_dir='./'
        ):
        super().__init__(
            K=K,
            save_results=save_results,
            save_dir=save_dir
        )
        self.n_workers = n_workers
        self.faiss_gpu = faiss_gpu
        

    def nearest_search(
            self, 
            embeddings, 
            labels, 
            file_names=None, 
            dataset_name='dataset'
        ):
        
        top_k = max(self.K)
        vector_dimension = embeddings.shape[1]
        faiss.omp_set_num_threads(self.n_workers)

        index = faiss.IndexFlatIP(vector_dimension)
        if self.faiss_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        distances, best_top_n_idxs = index.search(
            embeddings, 
            k=top_k+1
        )

        best_top_n_idxs = best_top_n_idxs[:, 1:]
        distances = distances[:, 1:]

        df_nearest = self.get_nearest_info(
            labels,
            best_top_n_idxs,
            distances,
            file_names=file_names
        )

        if self.save_results:
            df_nearest.to_feather(self.save_dir / f'{dataset_name}_top{top_k}.feather')

        return df_nearest