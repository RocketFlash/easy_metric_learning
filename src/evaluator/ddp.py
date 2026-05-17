from tqdm.auto import tqdm
import torch
import numpy as np
from .base import BaseEvaluator


class DDPEvaluator(BaseEvaluator):
    def __init__(
        self,
        config,
        model,
        save_dir="./",
        device="cpu",
        accelerator=None,
        is_eval=True,
        pca=None,
    ):
        device = getattr(accelerator, "device", device)
        super().__init__(
            config=config,
            model=model,
            save_dir=save_dir,
            device=device,
            is_eval=is_eval,
            pca=pca,
        )
        self.accelerator = accelerator

    def evaluate(self, data_info):
        if not getattr(data_info, "_ddp_prepared", False):
            data_info.dataloader = self.accelerator.prepare(data_info.dataloader)
            data_info._ddp_prepared = True

        embeddings, labels, file_names = self.generate_embeddings(data_info)
        if self.pca is not None:
            embeddings = self.pca.transform(embeddings)

        metrics = {}
        if self.accelerator.is_local_main_process:
            df_nearest = self.knn_algo.nearest_search(
                embeddings=embeddings,
                labels=labels,
                file_names=file_names,
                dataset_name=data_info.dataset_name,
            )
            metrics = self.calculate_metrics(
                df_nearest, dataset_name=data_info.dataset_name
            )

        return metrics

    def _gather_file_names(self, file_names, batch_size):
        if file_names is None:
            return [None] * batch_size

        if isinstance(file_names, (str, bytes)):
            file_names = [file_names]
        else:
            file_names = list(file_names)

        if getattr(self.accelerator, "num_processes", 1) > 1:
            try:
                file_names = self.accelerator.gather_for_metrics(
                    file_names, use_gather_object=True
                )
            except TypeError:
                try:
                    from accelerate.utils import gather_object

                    file_names = gather_object(file_names)
                except Exception:
                    raise RuntimeError("Failed to gather file names for DDP evaluation")

        file_names = list(file_names)
        if len(file_names) < batch_size:
            raise RuntimeError(
                f"Gathered {len(file_names)} file names for a DDP batch of {batch_size}"
            )

        return file_names[:batch_size]

    def generate_embeddings(self, data_info):
        embeddings = np.zeros(
            (data_info.dataset_stats.n_samples, self.config.embeddings_size),
            dtype=np.float32,
        )
        labels = np.zeros(data_info.dataset_stats.n_samples, dtype=object)
        file_names = np.zeros(data_info.dataset_stats.n_samples, dtype=object)
        n_missing_file_names = 0

        tqdm_test = tqdm(
            data_info.dataloader,
            total=int(len(data_info.dataloader)),
            disable=not self.accelerator.is_local_main_process,
        )
        index = 0

        with torch.no_grad():
            model_ = self.accelerator.unwrap_model(self.model)
            for batch_index, (images, targets, fnames) in enumerate(tqdm_test):
                if self.debug and batch_index >= 10:
                    break

                images = images.to(self.device)
                if hasattr(model_, "get_embeddings"):
                    output = model_.get_embeddings(images)
                else:
                    output = model_(images)

                output = self.accelerator.gather_for_metrics(output)
                targets = self.accelerator.gather_for_metrics(targets)

                if torch.is_tensor(output):
                    output = output.cpu().numpy()

                batch_size = min(output.shape[0], embeddings.shape[0] - index)
                if batch_size <= 0:
                    break

                output = output[:batch_size]
                targets = targets[:batch_size]
                lbls = [data_info.ids_to_labels[t] for t in targets.cpu().numpy()]
                batch_file_names = self._gather_file_names(fnames, batch_size)
                embeddings[index : (index + batch_size), :] = output
                labels[index : (index + batch_size)] = lbls
                file_names[index : (index + batch_size)] = batch_file_names
                n_missing_file_names += sum(name is None for name in batch_file_names)
                index += batch_size

        if index > 0 and n_missing_file_names == index:
            file_names = None

        if self.save_embeddings and self.accelerator.is_local_main_process:
            save_data = dict(embeddings=embeddings, labels=labels)
            if file_names is not None:
                save_data["file_names"] = file_names
            np.savez(
                self.save_dir / f"{data_info.dataset_name}_embeddings.npz", **save_data
            )

        return embeddings, labels, file_names
