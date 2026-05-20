from pathlib import Path

import numpy as np
import pandas as pd

from .base import BaseEvaluator
from .ddp import DDPEvaluator
from .verification import evaluate_verification_pairs


def _config_value(config, key, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _normalize(embeddings, eps=1e-12):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, eps)


def aggregate_templates(embeddings, template_ids, media_ids=None, normalize=True):
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if normalize:
        embeddings = _normalize(embeddings)
    template_ids = np.asarray(template_ids)
    if media_ids is None:
        media_ids = np.arange(len(template_ids))
    media_ids = np.asarray(media_ids)

    template_embeddings = []
    output_template_ids = []
    for template_id in sorted(np.unique(template_ids).tolist()):
        template_mask = template_ids == template_id
        template_media_ids = media_ids[template_mask]
        template_media_embeddings = embeddings[template_mask]

        media_embeddings = []
        for media_id in np.unique(template_media_ids).tolist():
            media_embeddings.append(
                template_media_embeddings[template_media_ids == media_id].mean(axis=0)
            )
        template_embedding = np.asarray(media_embeddings).mean(axis=0)
        template_embeddings.append(template_embedding)
        output_template_ids.append(template_id)

    template_embeddings = np.asarray(template_embeddings, dtype=np.float32)
    if normalize:
        template_embeddings = _normalize(template_embeddings)
    return output_template_ids, template_embeddings


def load_template_metadata(
    metadata_path,
    file_names,
    file_col="file_name",
    template_col="template_id",
    media_col="media_id",
):
    metadata_path = Path(metadata_path)
    if metadata_path.suffix.lower() == ".csv":
        metadata = pd.read_csv(metadata_path)
    else:
        metadata = pd.read_csv(
            metadata_path,
            sep=r"\s+",
            header=None,
            names=[file_col, template_col, media_col],
        )
    file_to_row = {str(row[file_col]): row for _, row in metadata.iterrows()}
    template_ids = []
    media_ids = []
    missing = []
    for file_name in file_names:
        file_name = str(file_name)
        basename = Path(file_name).name
        row = file_to_row.get(file_name, file_to_row.get(basename))
        if row is None:
            missing.append(file_name)
            continue
        template_ids.append(row[template_col])
        media_ids.append(row[media_col] if media_col in row else row[template_col])

    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"{len(missing)} files missing template metadata: {preview}")
    return np.asarray(template_ids), np.asarray(media_ids)


def load_template_pairs(
    pairs_path, template_id_to_index, left_col, right_col, label_col
):
    pairs_path = Path(pairs_path)
    if pairs_path.suffix.lower() == ".csv":
        pairs_df = pd.read_csv(pairs_path)
    else:
        pairs_df = pd.read_csv(
            pairs_path,
            sep=r"\s+",
            header=None,
            names=[left_col, right_col, label_col],
        )
    pairs = []
    is_same = []
    for row in pairs_df.itertuples(index=False):
        row_dict = row._asdict()
        left = row_dict[left_col]
        right = row_dict[right_col]
        pairs.append((template_id_to_index[left], template_id_to_index[right]))
        label = row_dict[label_col]
        is_same.append(str(label).lower() in {"1", "true", "t", "same", "yes"})
    return np.asarray(pairs, dtype=np.int64), np.asarray(is_same, dtype=bool)


class IJBTemplateEvaluatorMixin:
    def _init_ijb(self, config):
        evaluator_config = config.evaluation.evaluator
        self.save_results = _config_value(evaluator_config, "save_results", False)
        self.save_embeddings = _config_value(evaluator_config, "save_embeddings", False)
        self.fars = _config_value(evaluator_config, "fars", [1e-4, 1e-3, 1e-2])
        self.metadata_path = _config_value(evaluator_config, "metadata_path")
        self.pairs_path = _config_value(evaluator_config, "pairs_path")
        columns = _config_value(evaluator_config, "columns", {})
        self.file_col = _config_value(columns, "file", "file_name")
        self.template_col = _config_value(columns, "template", "template_id")
        self.media_col = _config_value(columns, "media", "media_id")
        self.left_col = _config_value(columns, "left", "template1")
        self.right_col = _config_value(columns, "right", "template2")
        self.label_col = _config_value(columns, "label", "is_same")

    def evaluate(self, data_info):
        embeddings, labels, file_names = self.generate_embeddings(data_info)
        if self.pca is not None:
            embeddings = self.pca.transform(embeddings)

        accelerator = getattr(self, "accelerator", None)
        if accelerator is not None and not accelerator.is_local_main_process:
            return {}

        template_ids, media_ids = load_template_metadata(
            self.metadata_path,
            file_names,
            file_col=self.file_col,
            template_col=self.template_col,
            media_col=self.media_col,
        )
        output_template_ids, template_embeddings = aggregate_templates(
            embeddings,
            template_ids,
            media_ids=media_ids,
            normalize=True,
        )
        template_id_to_index = {
            template_id: index for index, template_id in enumerate(output_template_ids)
        }
        pairs, is_same = load_template_pairs(
            self.pairs_path,
            template_id_to_index,
            self.left_col,
            self.right_col,
            self.label_col,
        )
        metrics = evaluate_verification_pairs(
            template_embeddings,
            pairs,
            is_same,
            fars=self.fars,
            normalize=False,
        )
        metrics["n_templates"] = len(output_template_ids)

        if self.save_results:
            df_metrics = pd.DataFrame(metrics.items(), columns=["metric", "score"])
            df_metrics.to_csv(
                self.save_dir / f"{data_info.dataset_name}_ijb_template_metrics.csv",
                index=False,
            )
        return metrics


class IJBTemplateEvaluator(IJBTemplateEvaluatorMixin, BaseEvaluator):
    def __init__(
        self,
        config,
        model=None,
        save_dir="./",
        device="cpu",
        model_info=None,
        is_eval=True,
        pca=None,
    ):
        super().__init__(
            config=config,
            model=model,
            save_dir=save_dir,
            device=device,
            model_info=model_info,
            is_eval=False,
            pca=pca,
        )
        self._init_ijb(config)


class DDPIJBTemplateEvaluator(IJBTemplateEvaluatorMixin, DDPEvaluator):
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
        super().__init__(
            config=config,
            model=model,
            save_dir=save_dir,
            device=device,
            accelerator=accelerator,
            is_eval=False,
            pca=pca,
        )
        self._init_ijb(config)
