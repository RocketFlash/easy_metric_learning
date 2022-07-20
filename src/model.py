import torch
import torch.nn as nn
from torch.cuda import amp
from torch.nn import Parameter
import torch.nn.functional as F

import timm
import math

from .margin import get_margin
from .backbone import get_backbone

class GeM(nn.Module):
    """GeM pooling: https://arxiv.org/pdf/1711.02512.pdf """
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class EmbeddigsNet(nn.Module):
    """
    A class for embeddings learning model  
    ...
    Attributes
    ----------
    model_name : 
        backbone name from timm library e.g. resnet50, resnext101_32x4d, efficientnet_b0, vit, swin etc
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
                       pool_type='avg',
                       dropout=0.0):
        super(EmbeddigsNet, self).__init__()

        if pool_type=='gem':
            self.pooling = GeM()
        elif pool_type=='avg':
            self.pooling = nn.AdaptiveAvgPool2d(1)
        else:
            self.pooling = None

        if 'iresnet' in model_name:
            self.backbone = get_backbone(model_name)
        else:
            self.backbone = timm.create_model(model_name, pretrained=pretrained, 
                                                          scriptable=True)
        
        if 'swin' in model_name or 'vit' in model_name:
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Identity()
            self.pooling = None
        elif 'efficientnet' in model_name:
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
            self.backbone.global_pool = nn.Identity()
        else:
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.backbone.global_pool = nn.Identity()


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
        return self.extract_features(x)


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


def get_model_embeddings(model_config=None,
                         model_name='efficientnet_b0', 
                         embeddings_size=512, 
                         pool_type='avg',
                         pretrained=True, 
                         dropout=0.0):
    if model_config is not None:
        model_name      = model_config['ENCODER_NAME'] 
        margin_type     = model_config['MARGIN_TYPE']
        embeddings_size = model_config['EMBEDDINGS_SIZE']   
        dropout         = model_config['DROPOUT_PROB']
        pool_type       = model_config['POOL_TYPE']
    model = EmbeddigsNet(model_name=model_name, 
                         embeddings_size=embeddings_size, 
                         pool_type=pool_type,
                         pretrained=pretrained, 
                         dropout=dropout)
    return model


def get_model(model_config=None,
              model_name='efficientnet_b0', 
              margin_type='arcface',
              pool_type='avg',
              embeddings_size=512, 
              pretrained=True, 
              dropout=0.0,
              out_features=1000,
              s=0.1,
              m=100,
              K=1,
              easy_margin=False,
              ls_eps=0.0):

    if model_config is not None:
        model_name      = model_config['ENCODER_NAME'] 
        margin_type     = model_config['MARGIN_TYPE']
        embeddings_size = model_config['EMBEDDINGS_SIZE']   
        dropout         = model_config['DROPOUT_PROB']
        out_features    = model_config['N_CLASSES']
        pool_type       = model_config['POOL_TYPE']
        s               = model_config['S']
        m               = model_config['M']
        K               = model_config['K']
        easy_margin     = model_config['EASY_MARGIN']
        ls_eps          = model_config['LS_PROB']

    embeddings_model = get_model_embeddings(model_name=model_name, 
                                            pool_type=pool_type,
                                            embeddings_size=embeddings_size, 
                                            pretrained=pretrained, 
                                            dropout=dropout)
    
    margin = get_margin(margin_type=margin_type,
                        embeddings_size=embeddings_size,
                        out_features=out_features,
                        s=s,
                        m=m,
                        K=K,
                        easy_margin=easy_margin,
                        ls_eps=ls_eps)
    
    model = MLNet(embeddings_model, margin)
    return model