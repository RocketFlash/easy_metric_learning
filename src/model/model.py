import torch
import torch.nn as nn
from .backbone import get_backbone
from .head import get_head


class EmbeddigsNet(nn.Module):
    """
    A class for embeddings learning model  
    
    Args:
        config_backbone : 
            config with backbone parameters 
        config_head : 
            config with head parameters
    """
    def __init__(
            self, 
            config_backbone, 
            config_head, 
        ):
        super(EmbeddigsNet, self).__init__()
        backbone, backbone_out_feats = get_backbone(config_backbone)
        self.head = get_head(config_head, backbone_out_feats)
        self.backbone = backbone

    
    def get_embeddings(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


    def forward(self, x):
        return self.get_embeddings(x)


class MLNet(nn.Module):
    """
    A class for metric learning model  
    
    Args:
        embeddings_net
            model generating embeddings
        margin
            margin module
    """
    def __init__(
            self, 
            embeddings_net, 
            margin
        ):
        super(MLNet, self).__init__()

        self.embeddings_net = embeddings_net
        self.margin = margin
        

    def get_embeddings(self, x):
        return self.embeddings_net(x)


    def forward(self, x, label):
        x_embedd = self.get_embeddings(x)
        x_margin = self.margin(x_embedd, label)
        return x_margin


def get_model_embeddings(
        config_backbone, 
        config_head
    ):

    model = EmbeddigsNet(
        config_backbone=config_backbone, 
        config_head=config_head
    )
    return model