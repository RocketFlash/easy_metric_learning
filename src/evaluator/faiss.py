import faiss
from .base import BaseEvaluator


class FAISSEvaluator(BaseEvaluator):
    def __init__(
            self, 
            config,
            model, 
            save_dir='./', 
            device='cpu', 
        ):
        super().__init__(
            config=config,
            model=model, 
            save_dir=save_dir, 
            device=device, 
        )
    

    def nearest_search(
            self, 
            embeddings, 
            labels, 
            file_names=None, 
            dataset_name='dataset'
        ):
        self.model.eval()
            
        vector_dimension = embeddings.shape[1]
        faiss.omp_set_num_threads(self.config.n_workers)

        index = faiss.IndexFlatIP(vector_dimension)
        if self.config.evaluation.evaluator.faiss_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        distances, best_top_n_idxs = index.search(
            embeddings, 
            k=self.K+1
        )

        best_top_n_idxs = best_top_n_idxs[:, 1:]
        distances = distances[:, 1:]

        df_nearest = self.get_nearest_info(
            labels,
            file_names,
            best_top_n_idxs,
            distances
        )

        if self.save_results:
            df_nearest.to_feather(self.save_dir / f'{dataset_name}_top{self.K}.feather')

        return df_nearest

                
            