from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class PCATransformer:
    components_: np.ndarray
    mean_: np.ndarray
    explained_variance_: Optional[np.ndarray] = None
    whiten: bool = False

    def transform(self, embeddings):
        embeddings = np.asarray(embeddings)
        transformed = (embeddings - self.mean_) @ self.components_.T
        if self.whiten:
            if self.explained_variance_ is None:
                raise ValueError("PCA whitening requires explained variance")
            explained_variance = np.asarray(self.explained_variance_)
            if not np.all(np.isfinite(explained_variance)) or np.any(
                explained_variance <= 0
            ):
                raise ValueError(
                    "PCA whitening requires positive finite explained variance"
                )
            transformed = transformed / np.sqrt(explained_variance)
        return transformed.astype(np.float32, copy=False)


def save_pca(pca, path):
    path = Path(path)
    data = {
        "components_": pca.components_,
        "mean_": pca.mean_,
        "whiten": bool(getattr(pca, "whiten", False)),
    }
    for key in [
        "explained_variance_",
        "singular_values_",
        "n_components_",
        "n_features_in_",
        "n_samples_",
    ]:
        value = getattr(pca, key, None)
        if value is not None:
            data[key] = value

    np.savez(path, **data)


def load_pca(path):
    path = Path(path)
    if path.suffix == ".pkl":
        raise ValueError(
            f"Refusing to load pickle PCA artifact {path}. "
            "Regenerate it with tools/dimensionality_reduction/train_pca.py to create a .npz artifact."
        )

    with np.load(path, allow_pickle=False) as data:
        explained_variance = (
            data["explained_variance_"] if "explained_variance_" in data.files else None
        )
        whiten = bool(data["whiten"]) if "whiten" in data.files else False
        return PCATransformer(
            components_=data["components_"],
            mean_=data["mean_"],
            explained_variance_=explained_variance,
            whiten=whiten,
        )
