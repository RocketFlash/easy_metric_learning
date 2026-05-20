import faiss
from .base import BaseKNN
from src.evaluator.rerank import is_rerank_enabled, rerank_from_config


class FAISSKNN(BaseKNN):
    def __init__(
        self,
        K=1,
        n_workers=8,
        save_results=False,
        faiss_gpu=False,
        save_dir="./",
        index_type="flat",
        nlist=4096,
        nprobe=32,
        hnsw_m=32,
        rerank_config=None,
    ):
        super().__init__(
            K=K,
            save_results=save_results,
            save_dir=save_dir,
            rerank_config=rerank_config,
        )
        self.n_workers = n_workers
        self.faiss_gpu = faiss_gpu
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.hnsw_m = hnsw_m

    def _build_index(self, embeddings):
        vector_dimension = embeddings.shape[1]
        index_type = self.index_type.lower()

        if index_type == "flat":
            index = faiss.IndexFlatIP(vector_dimension)
        elif index_type == "ivf":
            nlist = min(int(self.nlist), max(1, embeddings.shape[0]))
            quantizer = faiss.IndexFlatIP(vector_dimension)
            index = faiss.IndexIVFFlat(
                quantizer,
                vector_dimension,
                nlist,
                faiss.METRIC_INNER_PRODUCT,
            )
            index.train(embeddings)
            index.nprobe = min(int(self.nprobe), nlist)
        elif index_type == "hnsw":
            index = faiss.IndexHNSWFlat(
                vector_dimension,
                int(self.hnsw_m),
                faiss.METRIC_INNER_PRODUCT,
            )
        else:
            raise ValueError(f"Unknown FAISS index_type: {self.index_type}")

        if self.faiss_gpu:
            if index_type == "hnsw":
                raise ValueError("faiss_gpu is not supported for HNSW indexes")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        return index

    def nearest_search(
        self, embeddings, labels, file_names=None, dataset_name="dataset"
    ):

        top_k = max(self.K)
        if is_rerank_enabled(self.rerank_config):
            best_top_n_idxs, distances = rerank_from_config(
                embeddings, top_k=top_k, config=self.rerank_config
            )
            df_nearest = self.get_nearest_info(
                labels, best_top_n_idxs, distances, file_names=file_names
            )
            if self.save_results:
                df_nearest.to_feather(
                    self.save_dir / f"{dataset_name}_top{top_k}.feather"
                )
            return df_nearest

        faiss.omp_set_num_threads(self.n_workers)

        embeddings = embeddings.astype("float32", copy=True)
        faiss.normalize_L2(embeddings)
        index = self._build_index(embeddings)
        index.add(embeddings)
        distances, best_top_n_idxs = index.search(embeddings, k=top_k + 1)

        best_top_n_idxs = best_top_n_idxs[:, 1:]
        distances = distances[:, 1:]

        df_nearest = self.get_nearest_info(
            labels, best_top_n_idxs, distances, file_names=file_names
        )

        if self.save_results:
            df_nearest.to_feather(self.save_dir / f"{dataset_name}_top{top_k}.feather")

        return df_nearest
