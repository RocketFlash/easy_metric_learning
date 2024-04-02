from tqdm.auto import tqdm
import torch
import numpy as np
from .base import BaseEvaluator


class DDPEvaluator(BaseEvaluator):
    def __init__(
            self, 
            config,
            model, 
            save_dir='./', 
            accelerator=None,
            is_eval=True,
            pca=None
        ):
        super().__init__(
            config=config,
            model=model, 
            save_dir=save_dir, 
            is_eval=is_eval,
            pca=pca
        )
        self.accelerator = accelerator


    def evaluate(self, data_info):
        data_info.dataloader = self.accelerator.prepare(data_info.dataloader)
        embeddings, labels = self.generate_embeddings(data_info)
        if self.pca is not None:
            embeddings = self.pca.transform(embeddings)

        metrics = {}
        if self.accelerator.is_local_main_process:
            df_nearest = self.knn_algo.nearest_search(
                embeddings=embeddings, 
                labels=labels, 
                dataset_name=data_info.dataset_name
            )
            metrics = self.calculate_metrics(
                df_nearest, 
                dataset_name=data_info.dataset_name
            )

        return metrics
    

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
            data_info.dataloader, 
            total=int(len(data_info.dataloader)),
            disable=not self.accelerator.is_local_main_process
        )
        index = 0
        
        with torch.no_grad():
            for batch_index, (images, targets, fnames) in enumerate(tqdm_test):
                if self.debug and batch_index>=10: break
                
                model_ = self.accelerator.unwrap_model(self.model)
                if hasattr(model_, "get_embeddings"):
                    output = model_.get_embeddings(images)
                else:
                    output = model_(images)
                    
                output = self.accelerator.gather_for_metrics(output)
                targets = self.accelerator.gather_for_metrics(targets)
                
                if torch.is_tensor(output):
                    output = output.cpu().numpy()
                
                batch_size = output.shape[0]
                lbls = [data_info.ids_to_labels[t] for t in targets.cpu().numpy()]
                embeddings[index:(index+batch_size), :] = output
                labels[index:(index+batch_size)] = lbls
                index += batch_size

        if self.save_embeddings and self.accelerator.is_local_main_process:
            np.savez(
                self.save_dir / f'{data_info.dataset_name}_embeddings.npz', 
                embeddings=embeddings, 
                labels=labels,
            )

        return embeddings, labels
                
            