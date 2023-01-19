from .m_per_class import MPerClassSampler

def get_sampler(sampler_name='default', labels=[], m=5, batch_size=None, length_before_new_iter=1000000):
    if sampler_name=='default':
        return None

    return MPerClassSampler(labels, m, batch_size, length_before_new_iter)