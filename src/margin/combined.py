import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import math


class CombinedMargin(nn.Module):

    def __init__(self,
                 in_features, 
                 out_features,
                 m1=1.0,
                 m=0.5,
                 m3=0.0,
                 s=64.0,
                 ls_eps=0.0,
                 easy_margin=False):
        super(CombinedMargin, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.m1 = m1
        self.m2 = m
        self.m3 = m3
        self.m = m
        self.s = s
        self.easy_margin = easy_margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.ls_eps = ls_eps
        nn.init.xavier_uniform_(self.weight)
        '''
        COM(θ) = cos(m_1*θ+m_2) - m_3
        '''
        self.cos_m2 = math.cos(self.m2)
        self.sin_m2 = math.sin(self.m2)
        self.threshold = math.cos(math.pi - self.m2)


    def forward(self, x, gt_labels):
        x = F.normalize(x, dim=1)
        weights = F.normalize(self.weight, dim=1)
        cos_t = F.linear(x, weights)
        sin_t = torch.sqrt(1.0 - torch.pow(cos_t, 2))
        cos_tm = cos_t * self.cos_m2 - sin_t * self.sin_m2 - self.m3
        if self.easy_margin:
            # theta < pi/2, use cos(theta+m), else cos(theta)
            cos_tm = torch.where(cos_t > 0, cos_tm, cos_t)
        else:
            # theta + m < pi, use cos(theta+m), else cos(theta) - sin(theta)*m
            cos_tm = torch.where(cos_t > self.threshold, cos_tm, cos_t - self.m3 - sin_t * self.m2)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cos_theta.size(), device='cuda')
        one_hot = torch.zeros(cos_t.size()).to(x.device)
        one_hot.scatter_(1, gt_labels.view(-1, 1).long(), 1)

        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features

        output = (one_hot * cos_tm) + ((1.0 - one_hot) * cos_t)
        output *= self.s
        return output