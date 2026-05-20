from .base import BaseDataset
from .mxdataset import MXDataset


def get_dataset(root_dir, df_annos, transform, labels_to_ids, dataset_config):
    dataset_type = dataset_config.type

    if dataset_type == "mxnet":
        dataset = MXDataset(
            root_dir=root_dir,
            transform=transform,
            use_cache=dataset_config.use_cache,
            calc_cl_count=dataset_config.calc_cl_count,
        )
    else:
        dataset = BaseDataset(
            root_dir=root_dir,
            df_annos=df_annos,
            transform=transform,
            labels_to_ids=labels_to_ids,
            label_column=dataset_config.label_column,
            fname_column=dataset_config.fname_column,
            use_bboxes=dataset_config.use_bboxes,
            keypoints_column=getattr(dataset_config, "keypoints_column", None),
            num_keypoints=getattr(dataset_config, "num_keypoints", None),
            group_column=getattr(dataset_config, "group_column", None),
        )

    return dataset
