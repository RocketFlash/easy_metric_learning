from .simple import MLDataset
from .mxdataset import MXDataset


def get_dataset(
        root_dir, 
        df_annos, 
        transform,
        labels_to_ids,
        dataset_config
    ):
    dataset_type = dataset_config.type

    if dataset_type=='mxnet':
        dataset = MXDataset(root_dir=root_dir, 
                            transform=transform,
                            use_cache=dataset_config.use_cache,
                            calc_cl_count=dataset_config.calc_cl_count)
    else:
        dataset = MLDataset(
            root_dir=root_dir, 
            df_annos=df_annos,     
            transform=transform,
            labels_to_ids=labels_to_ids,
            label_column=dataset_config.label_column,
            fname_column=dataset_config.fname_column,
            use_bboxes=dataset_config.use_bboxes
        )

    return dataset