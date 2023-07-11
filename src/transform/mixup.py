import torch
import numpy as np

def mixup(data, targets, alpha):
    # Code from https://www.kaggle.com/code/riadalmadani/fastai-effb0-base-model-birdclef2023
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    new_data = data * lam + shuffled_data * (1 - lam)
    new_targets = [targets, shuffled_targets, lam]
    return new_data, new_targets