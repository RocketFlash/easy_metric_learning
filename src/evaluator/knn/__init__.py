from .base import BaseKNN
from .faiss import FAISSKNN


def get_knn_search(config_knn, K, save_dir):
    if config_knn.type == "faiss":
        return FAISSKNN(
            K=K,
            n_workers=config_knn.n_workers,
            save_results=config_knn.save_results,
            faiss_gpu=config_knn.faiss_gpu,
            save_dir=save_dir,
            index_type=getattr(config_knn, "index_type", "flat"),
            nlist=getattr(config_knn, "nlist", 4096),
            nprobe=getattr(config_knn, "nprobe", 32),
            hnsw_m=getattr(config_knn, "hnsw_m", 32),
            rerank_config=getattr(config_knn, "rerank", None),
        )
    else:
        return BaseKNN(
            K=K,
            n_chunks=getattr(config_knn, "n_chunks", 5),
            save_results=config_knn.save_results,
            save_dir=save_dir,
            rerank_config=getattr(config_knn, "rerank", None),
        )
