import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import math


class ArcMarginProduct(nn.Module):
    """
    Implementation of arcmargin 
    Implementation taken from: https://github.com/wujiyang/Face_Pytorch/blob/master/margin/ArcMarginProduct.py 
    ...
    Attributes
    ----------
    in_features : 
        number of input features
    out_features : 
        number of output features 
    s : 
        norm of input feature 
    m :
        margin cos(theta + m)
    easy_margin:
        use easy margin approach
    ls_eps :
        label smoothing
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin

        if isinstance(m, dict):
            self.cos_m, self.sin_m = {}, {}
            self.th, self.mm = {}, {}
            for class_id, mrgn in m.items():
                self.cos_m[class_id] = math.cos(mrgn)
                self.sin_m[class_id] = math.sin(mrgn)

                self.th[class_id] = math.cos(math.pi - mrgn)
                self.mm[class_id] = math.sin(math.pi - mrgn) * mrgn
        else:
            self.cos_m, self.sin_m = math.cos(m), math.sin(m)

            # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
            self.th = math.cos(math.pi - m)
            self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        if isinstance(self.m, dict):
            cos_m_i, sin_m_i, th_i, mm_i = [], [], [], []
            for l in label:
                l = l.detach().item()
                cos_m_i.append(self.cos_m[l])
                sin_m_i.append(self.sin_m[l])
                th_i.append(self.th[l])
                mm_i.append(self.mm[l])
            cos_m_i = torch.Tensor(cos_m_i).to(label.device)
            sin_m_i = torch.Tensor(sin_m_i).to(label.device)
            th_i = torch.Tensor(th_i).to(label.device)
            mm_i = torch.Tensor(mm_i).to(label.device)
        else:
            cos_m_i = self.cos_m
            sin_m_i = self.sin_m
            th_i = self.th
            mm_i = self.mm

        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))

        if isinstance(self.m, dict):
            cos_m_i = torch.unsqueeze(cos_m_i, 1)
            cos_m_i = cos_m_i.repeat(1, cosine.shape[1])
            sin_m_i = torch.unsqueeze(sin_m_i, 1)
            sin_m_i = sin_m_i.repeat(1, cosine.shape[1])
            th_i = torch.unsqueeze(th_i, 1)
            th_i = th_i.repeat(1, cosine.shape[1])
            mm_i = torch.unsqueeze(mm_i, 1)
            mm_i = mm_i.repeat(1, cosine.shape[1])

        phi = cosine * cos_m_i - sine * sin_m_i

        if self.easy_margin:
            phi = torch.where(cosine.float() > 0, phi.float(), cosine.float())
        else:
            phi = torch.where(cosine.float() > th_i, phi.float(), cosine.float() - mm_i)

        one_hot = torch.zeros_like(cosine)
        if isinstance(label, list):
            label1, label2, lam = label
            one_hot.scatter_(1, label1.view(-1, 1).long(), lam)
            one_hot.scatter_(1, label2.view(-1, 1).long(), (1 - lam))
        else:
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features

        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output

    
    def update(self, m=0.5):
        self.m = m
        if isinstance(m, dict):
            self.cos_m, self.sin_m = {}, {}
            self.th, self.mm = {}, {}
            for class_id, mrgn in m.items():
                self.cos_m[class_id] = math.cos(mrgn)
                self.sin_m[class_id] = math.sin(mrgn)

                self.th[class_id] = math.cos(math.pi - mrgn)
                self.mm[class_id] = math.sin(math.pi - mrgn) * mrgn
        else:
            self.cos_m, self.sin_m = math.cos(m), math.sin(m)

            # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
            self.th = math.cos(math.pi - m)
            self.mm = math.sin(math.pi - m) * m


class AddMarginProduct(nn.Module):
    """Large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    """

    def __init__(self, in_features, out_features, ls_eps=0, s=30.0, m=0.40):
        super(AddMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ls_eps = ls_eps  # label smoothing
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)


    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
       
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  
        output *= self.s

        return output


    def update(self, m=0.5):
        self.m = m