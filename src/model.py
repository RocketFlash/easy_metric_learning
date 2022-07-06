import torch
import torch.nn as nn
from torch.cuda import amp
from torch.nn import Parameter
import torch.nn.functional as F

import timm
import math

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

class AdaFace(nn.Module):
    """
    Implementation of adaface
    Implementation taken from: https://github.com/mk-minchul/AdaFace 
    """
    def __init__(self,
                 in_features=512,
                 out_features=70722,
                 m=0.4,
                 h=0.333,
                 s=64.,
                 t_alpha=1.0,
                 use_batchnorm=False):
        super(AdaFace, self).__init__()
        self.classnum = out_features
        self.kernel = Parameter(torch.Tensor(in_features, out_features))

        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m 
        self.eps = 1e-3
        self.h = h
        self.s = s
        self.use_batchnorm = use_batchnorm
        self.t_alpha = t_alpha

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


class SubcenterArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, K=3, s=30.0, m=0.50, easy_margin=False, ls_eps=0.0):
        super(SubcenterArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.K = K
        self.ls_eps = ls_eps
        self.weight = Parameter(torch.FloatTensor(out_features*self.K, in_features))
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
        
        if self.K > 1:
            cosine = torch.reshape(cosine, (-1, self.out_features, self.K))
            cosine, _ = torch.max(cosine, axis=2)
        
        if isinstance(self.m, dict):
            cos_m_i = torch.unsqueeze(cos_m_i, 1)
            cos_m_i = cos_m_i.repeat(1, cosine.shape[1])
            sin_m_i = torch.unsqueeze(sin_m_i, 1)
            sin_m_i = sin_m_i.repeat(1, cosine.shape[1])
            th_i = torch.unsqueeze(th_i, 1)
            th_i = th_i.repeat(1, cosine.shape[1])
            mm_i = torch.unsqueeze(mm_i, 1)
            mm_i = mm_i.repeat(1, cosine.shape[1])
            

        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * cos_m_i - sine * sin_m_i

        if self.easy_margin:
            phi = torch.where(cosine.float() > 0, phi.float(), cosine.float())
        else:
            phi = torch.where(cosine.float() > th_i, phi.float(), cosine.float() - mm_i)

        one_hot = torch.zeros_like(cosine)
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
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super(AddMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)


    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device='cuda')
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output


    def update(self, m=0.5):
        self.m = m


class AdaCos(nn.Module):
    def __init__(self, in_features, out_features, m=0.50, ls_eps=0, theta_zero=math.pi/4):
        super(AdaCos, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.theta_zero = theta_zero
        self.s = math.log(out_features - 1) / math.cos(theta_zero)
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)


    def forward(self, input, label):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.weight)
        # dot product
        logits = F.linear(x, W)
        # add margin
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(theta + self.m)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        output = logits * (1 - one_hot) + target_logits * one_hot
        # feature re-scale
        with torch.no_grad():
            B_avg = torch.where(one_hot < 1, torch.exp(self.s * logits.float()), torch.zeros_like(logits.float()))
            B_avg = torch.sum(B_avg) / input.size(0)
            theta_med = torch.median(theta)
            self.s = torch.log(B_avg) / torch.cos(torch.min(self.theta_zero * torch.ones_like(theta_med), theta_med))
        output *= self.s

        return output

    
    def update(self, m=0.5):
        self.m = m


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
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

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


class EmbeddigsNet(nn.Module):
    """
    A class for embeddings learning model  
    ...
    Attributes
    ----------
    model_name : 
        backbone name e.g. resnet50, resnext101_32x4d, efficientnet_b0 etc
    embeddings_size : 
        model output size (vector size in embeddings space) 
    DP : 
        DataParallel mode
    pretrained:
        load pretrained weights
    dropout:
        probability in dropout layers
    """
    def __init__(self, model_name='efficientnet_b0', 
                       embeddings_size=512, 
                       pretrained=True,
                       dropout=0.0):
        super(EmbeddigsNet, self).__init__()

        self.backbone = timm.create_model(model_name, pretrained=pretrained, 
                                                      scriptable=True)
        
        if 'swin' in model_name:
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Identity()
            self.pooling = None
        else:
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
            self.backbone.global_pool = nn.Identity()
            self.pooling = nn.AdaptiveAvgPool2d(1)

        self.dropout = nn.Dropout(p=dropout)
        self.n_features = embeddings_size

        self.classifier = nn.Linear(in_features, self.n_features)
        self.bn = nn.BatchNorm1d(self.n_features)
        self._init_params()

        
        # Unfreeze model weights
        for param in self.backbone.parameters():
            param.requires_grad = True

    
    def _init_params(self):
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)


    def extract_features(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        if self.pooling is not None:
            x = self.pooling(x).view(batch_size, -1)

        x = self.dropout(x)
        x = self.classifier(x)
        x = self.bn(x)

        return x


    def forward(self, x):
        x = self.extract_features(x)
        return x


class MLNet(nn.Module):
    """
    A class for metric learning model  
    ...
    Attributes
    ----------
    embeddings_net
        model generating embeddings
    margin
        margin module
    """
    def __init__(self, embeddings_net, margin):
        super(MLNet, self).__init__()

        self.embeddings_net = embeddings_net
        self.margin = margin


    def forward(self, x, label):
        x = self.embeddings_net(x)
        x = self.margin(x, label)
        return x


def get_model(model_name='efficientnet_b0', 
              margin_type='arcface',
              embeddings_size=512, 
              pretrained=True, 
              dropout=0.0,
              out_features=1000,
              scale_size=0.1,
              m=100,
              K=1,
              easy_margin=False,
              ls_eps=0.0):
    embeddings_model = EmbeddigsNet(model_name=model_name, 
                                    embeddings_size=embeddings_size, 
                                    pretrained=pretrained, 
                                    dropout=dropout)
    if margin_type=='adacos':
        margin = AdaCos(in_features=embeddings_size,
                        out_features=out_features, 
                        m=m,  
                        ls_eps=ls_eps)
    elif margin_type=='adaface_bn':
        margin = AdaFace(in_features=embeddings_size,
                         out_features=out_features,
                         m=m,
                         h=0.333,
                         s=scale_size,
                         t_alpha=1.0,
                         use_batchnorm=True)
    elif margin_type=='adaface':
        margin = AdaFace(in_features=embeddings_size,
                         out_features=out_features,
                         m=m,
                         h=0.333,
                         s=scale_size,
                         t_alpha=1.0)
    elif margin_type=='cosface':
        margin = AddMarginProduct(in_features=embeddings_size,
                                out_features=out_features, 
                                s=scale_size, 
                                m=m)
    elif margin_type=='subcenter_arcface':
        margin = SubcenterArcMarginProduct(in_features=embeddings_size,
                                out_features=out_features, 
                                s=scale_size, 
                                m=m,
                                K=K, 
                                easy_margin=easy_margin, 
                                ls_eps=ls_eps)
    else:
        margin = ArcMarginProduct(in_features=embeddings_size,
                                out_features=out_features, 
                                s=scale_size, 
                                m=m, 
                                easy_margin=easy_margin, 
                                ls_eps=ls_eps)
    model = MLNet(embeddings_model, margin)
    return model


def get_model_embeddings(model_name='efficientnet_b0', 
              embeddings_size=512, 
              pretrained=True, 
              dropout=0.0):
    model = EmbeddigsNet(model_name=model_name, 
                                    embeddings_size=embeddings_size, 
                                    pretrained=pretrained, 
                                    dropout=dropout)
    return model