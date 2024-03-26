from tqdm.auto import tqdm
import torch
import numpy as np
import pandas as pd
from ..metric.basic import recall_at_k
from .knn import get_knn_search


class BaseEvaluator:
    def __init__(
            self, 
            config,
            model=None, 
            save_dir='./', 
            device='cpu',
            model_info=None
        ):

        self.config = config
        self.model  = model
        self.device = device
        self.model_info = model_info

        if self.model_info is not None:
            self.model = self.model_info['model']
            if 'tf' in self.model_info['model_type']:
                import tensorflow as tf
            elif 'onnx' in self.model_info['model_type']:
                import onnxruntime as ort
    
        self.save_results = config.evaluation.evaluator.save_results
        self.save_embeddings = config.evaluation.evaluator.save_embeddings
        self.save_dir = save_dir
        self.save_dir.mkdir(exist_ok=True)
        self.K = config.evaluation.evaluator.K

        self.knn_algo = get_knn_search(
            config.evaluation.knn,
            K=self.K,
            save_dir=self.save_dir
        )

        self.debug = config.debug
        

    def evaluate(self, data_info):
        embeddings, labels, file_names = self.generate_embeddings(data_info)
        df_nearest = self.knn_algo.nearest_search(
            embeddings=embeddings, 
            labels=labels, 
            file_names=file_names, 
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

        file_names = np.zeros(
            data_info.dataset_stats.n_samples, 
            dtype=object
        )

        tqdm_test = tqdm(
            data_info.dataloader, 
            total=int(len(data_info.dataloader))
        )
        index = 0
        
        with torch.no_grad():
            for batch_index, (images, targets, fnames) in enumerate(tqdm_test):
                if self.debug and batch_index>=10: break
                
                if self.model_info is not None:
                    if self.model_info['model_type'] in ['torch', 'traced']:
                        images = images.to(self.device)
                        output = self.model(images)
                    elif self.model_info['model_type']=='onnx':
                        output = self.model.run( None, {"input": images.numpy()})[0]
                    elif self.model_info['model_type'] in ['tf_32', 'tf_16', 'tf_int', 'tf_dyn']:
                        data_np = torch.permute(images, (0, 2, 3, 1)).numpy()
                        tf_data = tf.convert_to_tensor(data_np)
                        output  = self.model(input=tf_data)['output']
                    elif self.model_info['model_type']=='tf_full_int':
                        data_np = torch.permute(images, (0, 2, 3, 1)).numpy()
                        input_scale, input_zero_point = self.model_info['input_details']["quantization"]
                        output_scale, output_zero_point = self.model_info['output_details']['quantization']
                        tf_sample_int8 = data_np / input_scale + input_zero_point
                        tf_sample_int8 = tf_sample_int8.astype(self.model_info['input_details']["dtype"])
                        tf_sample_int8 = tf.convert_to_tensor(tf_sample_int8)
                        output = self.model(input=tf_sample_int8)
                        output = output_scale * (output['output'].astype(np.float32) - output_zero_point)
                else:
                    images = images.to(self.device)
                    output = self.model.get_embeddings(images)

                if torch.is_tensor(output):
                    output = output.cpu().numpy()

                batch_size = output.shape[0]
                lbls = [data_info.ids_to_labels[t] for t in targets.cpu().numpy()]
                embeddings[index:(index+batch_size), :] = output
                labels[index:(index+batch_size)] = lbls
                file_names[index:(index+batch_size)] = fnames
                index += batch_size

        if self.save_embeddings:
            np.savez(
                self.save_dir / f'{data_info.dataset_name}_embeddings.npz', 
                embeddings=embeddings, 
                labels=labels,
                file_names=file_names
            )

        return embeddings, labels, file_names
    

    def calculate_metrics(self, df_nearest, dataset_name='dataset'):
        predictions  = np.array([row.astype(str) for row in df_nearest['prediction'].to_numpy()])
        similarities = np.array([row for row in df_nearest['similarity'].to_numpy()])
        gts          = df_nearest['gt'].to_numpy()

        metrics = {}
        
        for k in self.K:
            metrics[f'R@{k}'] = round(recall_at_k(gts, predictions, k=k), 5)

        df_metrics = pd.DataFrame(
            metrics.items(), 
            columns=['metric', 'score']
        )

        if self.save_results:
            df_metrics.to_csv(
                self.save_dir / f'{dataset_name}_metrics.csv', 
                index=False
            )

        return metrics

                
            