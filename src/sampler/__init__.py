from .m_per_class import MPerClassSampler
from .pk import PKSampler
from .advanced import ClassBalancedSampler, HardNegativeSampler, HierarchicalPKSampler


def get_sampler(
    labels,
    sampler_config,
    groups=None,
):

    sampler_type = sampler_config.type
    if sampler_type == "balanced":
        return MPerClassSampler(
            labels,
            sampler_config.m,
            sampler_config.batch_size,
            sampler_config.length_before_new_iter,
            seed=getattr(sampler_config, "seed", 0),
        )
    elif sampler_type == "pk":
        return PKSampler(
            labels,
            sampler_config.p,
            sampler_config.k,
            sampler_config.batch_size,
            sampler_config.length_before_new_iter,
            seed=getattr(sampler_config, "seed", 0),
        )
    elif sampler_type == "class_balanced":
        return ClassBalancedSampler(
            labels,
            beta=getattr(sampler_config, "beta", 0.9999),
            num_samples=getattr(sampler_config, "num_samples", None),
            replacement=getattr(sampler_config, "replacement", True),
            seed=getattr(sampler_config, "seed", 0),
        )
    elif sampler_type == "hierarchical_pk":
        groups = getattr(sampler_config, "groups", groups)
        return HierarchicalPKSampler(
            labels,
            groups=groups,
            groups_per_batch=sampler_config.groups_per_batch,
            labels_per_group=sampler_config.labels_per_group,
            samples_per_label=sampler_config.samples_per_label,
            length_before_new_iter=sampler_config.length_before_new_iter,
            seed=getattr(sampler_config, "seed", 0),
        )
    elif sampler_type == "hard_negative":
        embeddings = getattr(sampler_config, "embeddings", None)
        embeddings_path = getattr(sampler_config, "embeddings_path", None)
        if embeddings is None and embeddings_path is not None:
            import numpy as np

            loaded = np.load(embeddings_path)
            embeddings = loaded["embeddings"] if hasattr(loaded, "files") else loaded
        if embeddings is None:
            raise ValueError(
                "hard_negative sampler requires embeddings or embeddings_path"
            )
        return HardNegativeSampler(
            labels,
            embeddings=embeddings,
            p=sampler_config.p,
            k=sampler_config.k,
            hard_negative_rate=sampler_config.hard_negative_rate,
            length_before_new_iter=sampler_config.length_before_new_iter,
            seed=getattr(sampler_config, "seed", 0),
        )
    else:
        return None
