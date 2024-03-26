import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import math

from .utils import l2_norm

class AdaFace(nn.Module):
    """
    Implementation of adaface
    Implementation taken from: https://github.com/mk-minchul/AdaFace 
    """
    def __init__(
            self,
            in_features=512,
            out_features=70722,
            m=0.4,
            h=0.333,
            s=64.,
            t_alpha=1.0,
            ls_eps=0.0,
            use_batchnorm=False
        ):
        super(AdaFace, self).__init__()
        self.classnum = out_features
        self.in_features = in_features
        self.out_features = out_features
        self.kernel = Parameter(torch.Tensor(in_features, out_features))

        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m 
        self.eps = 1e-3
        self.h = h
        self.s = s
        self.use_batchnorm = use_batchnorm
        self.t_alpha = t_alpha
        self.ls_eps = ls_eps

        if self.use_batchnorm:
            self.norm_layer = nn.BatchNorm1d(1, eps=self.eps, momentum=self.t_alpha, affine=False)
        else:
            self.register_buffer('t', torch.zeros(1))
            self.register_buffer('batch_mean', torch.ones(1)*(20))
            self.register_buffer('batch_std', torch.ones(1)*100)


    def forward(self, x, label):

        norms = torch.norm(x, 2, -1, keepdim=True)
        embbedings  = x / norms

        kernel_norm = l2_norm(self.kernel,axis=0)
        cosine = torch.mm(embbedings,kernel_norm)
        cosine = cosine.clamp(-1+self.eps, 1-self.eps) # for stability

        safe_norms = torch.clip(norms, min=0.001, max=100) # for stability
        safe_norms = safe_norms.clone().detach()

        if self.use_batchnorm:
            margin_scaler = self.norm_layer(safe_norms)
        else:
            # update batchmean batchstd
            with torch.no_grad():
                mean = safe_norms.mean().detach()
                std = safe_norms.std().detach()
                self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
                self.batch_std =  std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

            margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std+self.eps) # 66% between -1, 1
            margin_scaler = margin_scaler * self.h # 68% between -0.333 ,0.333 when h:0.333
            margin_scaler = torch.clip(margin_scaler, -1, 1)

        # g_angular
        m_arc = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_arc.scatter_(1, label.reshape(-1, 1), 1.0)

        if self.ls_eps > 0:
            m_arc = (1 - self.ls_eps) * m_arc + self.ls_eps / self.out_features

        g_angular = self.m * margin_scaler * -1
        m_arc = m_arc * g_angular
        theta = cosine.acos()
        theta_m = torch.clip(theta + m_arc, min=self.eps, max=math.pi-self.eps)
        cosine = theta_m.cos()

        # g_additive
        m_cos = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_cos.scatter_(1, label.reshape(-1, 1), 1.0)
        g_add = self.m + (self.m * margin_scaler)
        m_cos = m_cos * g_add
        cosine = cosine - m_cos

        # scale
        scaled_cosine_m = cosine * self.s
        return scaled_cosine_m