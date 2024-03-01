from tqdm.auto import tqdm
import torch
import numpy as np
import faiss
import time
from easydict import EasyDict as edict
from ..data import get_test_data_from_config


class MLEvaluator:
    def __init__(
            self, 
            config,
            model, 
            logger, 
            work_dir='./', 
            device='cpu', 
            ids_to_labels=None
        ):

        self.config = config
        self.model  = model
        self.logger = logger
        self.device = device
        self.work_dir = work_dir
        self.ids_to_labels = ids_to_labels
        
        self.debug = config.debug
        self.visualize_batch = config.visualize_batch

        self.data_infos = self.get_eval_datasets()
    

    def get_eval_datasets(self):
        return get_test_data_from_config(self.config)
    

    def generate_embeddings(self, data_info):
        embeddings = np.zeros(
            (data_info.dataset_stats.n_samples, 
                self.config.embeddings_size), 
            dtype=np.float32
        )
        labels = np.zeros(
            data_info.dataset_stats.n_samples, 
            dtype=object
        )

        tqdm_test = tqdm(
            data_info.test_loader, 
            total=int(len(data_info.test_loader))
        )
        index = 0
        
        with torch.no_grad():
            for batch_index, (images, targets) in enumerate(tqdm_test):
                images = images.to(self.device)
                output = self.model.get_embeddings(images)

                if torch.is_tensor(output):
                    output = output.cpu().numpy()

                batch_size = output.shape[0]
                lbls = [data_info.ids_to_labels[t] for t in targets.cpu().numpy()]
                embeddings[index:(index+batch_size), :] = output
                labels[index:(index+batch_size)] = lbls
                index += batch_size

        return embeddings, labels


    def test(self):
        self.model.eval()
            
        for data_info in self.data_infos:
            embeddings, labels = self.generate_embeddings(data_info)
            start = time.time()
            vector_dimension = embeddings.shape[1]
            faiss.omp_set_num_threads(self.config.n_workers)

            index = faiss.IndexFlatIP(vector_dimension)
            # if args.faiss_gpu:
            #     if args.verbose: print('Use FAISS GPU')
            #     res = faiss.StandardGpuResources()
            #     index = faiss.index_cpu_to_gpu(res, 0, index)

            faiss.normalize_L2(embeddings)
            index.add(embeddings)
            distances, best_top_n_idxs = index.search(
                embeddings, 
                k=self.config.evaluation.K
            )
            end = time.time()

            best_top_n_idxs = best_top_n_idxs[:, 1:]
            distances = distances[:, 1:]
                
            