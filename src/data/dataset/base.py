from pathlib import Path
import numpy as np
import cv2
import torch
import warnings
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from ..utils import get_labels_to_ids_map


class BaseDataset(Dataset):
    """Base metric learning Dataset. Read images, apply
       augmentation and preprocessing transformations.

    Args:
        root_dir (str): path to data folder
        df_annos (str): dataframe with names
        transform (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)

    """

    def __init__(
        self,
        root_dir,
        df_annos,
        transform=None,
        labels_to_ids=None,
        use_bboxes=False,
        label_column="label",
        fname_column="file_name",
    ):

        self.images_paths = []
        self.file_names = []
        self.labels = []
        self.bboxes = []
        self.use_bboxes = use_bboxes

        if not isinstance(root_dir, list) and not isinstance(df_annos, list):
            root_dir = [root_dir]
            df_annos = [df_annos]

        file_names = [df_nms[fname_column].tolist() for df_nms in df_annos]
        labels = [df_nms[label_column].tolist() for df_nms in df_annos]

        for idx in range(len(file_names)):
            file_names_i = file_names[idx]
            root_dir_i = Path(root_dir[idx])
            self.images_paths += [
                str(root_dir_i / str(fname)) for fname in file_names_i
            ]
            self.file_names += [str(fname) for fname in file_names_i]

        for labels_i in labels:
            self.labels += labels_i
        self.labels = np.array(self.labels, dtype=str)

        if self.use_bboxes:
            for df_nms in df_annos:
                if "bbox" in df_nms:
                    bboxes = df_nms["bbox"].tolist()
                    bboxes = [bbox.split(" ") for bbox in bboxes]
                    bboxes = [[int(b) for b in bbox] for bbox in bboxes]
                else:
                    bboxes = [None] * len(df_nms)
                self.bboxes += bboxes

        if labels_to_ids is None:
            labels_names = sorted(np.unique(self.labels).tolist())
            self.labels_to_ids, self.ids_to_labels = get_labels_to_ids_map(labels_names)
        else:
            self.labels_to_ids = labels_to_ids
            self.ids_to_labels = {v: k for k, v in self.labels_to_ids.items()}
        self.label_ids = np.array([self.labels_to_ids[l] for l in self.labels])

        self.transform = transform

    @staticmethod
    def _read_cv_image(image_path):
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(image_path)

        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if image.ndim != 3:
            raise ValueError(f"Unsupported image shape {image.shape}")

        if image.shape[2] == 1:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

        raise ValueError(f"Unsupported image shape {image.shape}")

    @staticmethod
    def _as_uint8(image):
        if image.dtype == np.uint8:
            return image

        if np.issubdtype(image.dtype, np.floating):
            image = np.clip(image, 0, 1) * 255
        else:
            image = np.clip(image, 0, 255)

        return image.round().astype(np.uint8)

    @staticmethod
    def _read_gif_image(image_path):
        image = plt.imread(image_path)

        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)

        elif image.ndim != 3:
            raise ValueError(f"Unsupported image shape {image.shape}")

        elif image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        elif image.shape[2] >= 3:
            image = image[:, :, :3]
        else:
            raise ValueError(f"Unsupported image shape {image.shape}")

        return BaseDataset._as_uint8(image)

    def _read_image(self, image_path):
        if Path(image_path).suffix.lower() == ".gif":
            return self._read_gif_image(image_path)

        return self._read_cv_image(image_path)

    def get_labels_to_ids(self):
        return self.labels_to_ids

    def get_ids_to_labels(self):
        return self.ids_to_labels

    def __getitem__(self, i):
        image_path = self.images_paths[i]
        file_name = self.file_names[i]

        try:
            image = self._read_image(image_path)
        except (
            FileNotFoundError,
            OSError,
            cv2.error,
            IndexError,
            TypeError,
            ValueError,
        ) as exc:
            warnings.warn(f"Corrupted image {image_path}: {exc}", RuntimeWarning)
            return None

        if self.use_bboxes:
            bbox = self.bboxes[i]
            if bbox is not None:
                x1, y1, w, h = bbox
                image = image[y1 : (y1 + h), x1 : (x1 + w), :]

        if self.transform:
            sample = self.transform(image=image)
            image = sample["image"]

        data = torch.tensor(self.label_ids[i], dtype=torch.long)

        return image, data, file_name

    def __len__(self):
        return len(self.images_paths)
