from .m_per_class import MPerClassSampler


def get_sampler(
        labels,
        sampler_config, 
    ):

    sampler_type = sampler_config.type
    if sampler_type=='balanced':
        MPerClassSampler(
            labels, 
            sampler_config.m, 
            sampler_config.batch_size, 
            sampler_config.length_before_new_iter
        )
    else:
        return None