import torch

def l2_norm(input_x,axis=1):
    norm = torch.norm(input_x,2,axis,True)
    output = torch.div(input_x, norm)
    return output