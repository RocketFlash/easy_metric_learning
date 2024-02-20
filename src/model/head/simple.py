import torch
import torch.nn as nn


class SimpleHead(nn.Module):
    """
    A class for simple head  
    
    Args:
        backbone_out_feats:
            backbone output size
        embeddings_size : 
            model output size (vector size in embeddings space) 
        pretrained:
            load pretrained weights
        dropout:
            probability in dropout layers
        freeze_backbone:
            if True then backbone weights become untrainable
    """
    def __init__(self, 
                 backbone_out_feats,
                 embeddings_size=512, 
                 dropout_p=0.0):
        super(SimpleHead, self).__init__()

        self.dropout = nn.Dropout(p=dropout_p)
        self.head = nn.Linear(backbone_out_feats, embeddings_size)
        self.bn = nn.BatchNorm1d(embeddings_size)
        self._init_params()


    def _init_params(self):
        nn.init.xavier_normal_(self.head.weight)
        nn.init.constant_(self.head.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)


    def forward(self, x):
        x = self.dropout(x)
        print('dropout:', x.shape)
        x = self.head(x)
        print('head:', x.shape)
        x = self.bn(x)
        print('bn:', x.shape)
        return x