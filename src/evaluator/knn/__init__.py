from .base import BaseKNN
from .faiss import FAISSKNN

def get_knn_search(
        config_knn,
        K,
        save_dir
    ):
    if config_knn.type=='faiss':
        return FAISSKNN(
            K=K,
            n_workers=config_knn.n_workers,
            save_results=config_knn.save_results,
            faiss_gpu=config_knn.faiss_gpu,
            save_dir=save_dir
        )
    else:
        return BaseKNN(
            K=K,
            save_results=config_knn.save_results,
            save_dir=save_dir
        )