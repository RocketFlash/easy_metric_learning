import torch
from torch.profiler import profile, record_function, ProfilerActivity
import pprint

def profile_model(model, sample):
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            with torch.no_grad():
                model(sample)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))


def test_model(model, sample, to_profile=False):
    if to_profile:
        profile_model(model, sample)
    with torch.no_grad():
        o_t_o = model(sample)
        pprint.pprint(o_t_o[:, :8])
        