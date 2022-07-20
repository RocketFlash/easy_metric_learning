import torch
import torch.nn as nn

class Softmax(nn.Module):
    def __init__(self, in_features, out_features):
        super(Softmax, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(self.in_features, self.out_features)

    def forward(self, x):
        return self.fc(x)