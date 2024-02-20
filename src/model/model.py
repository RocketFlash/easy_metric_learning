import torch
import torch.nn as nn
from .backbone import get_backbone
from .head import get_head


class EmbeddigsNet(nn.Module):
    """
    A class for embeddings learning model  
    
    Args:
        backbone_config : 
            config with backbone parameters 
        head_config : 
            config with head parameters
    """
    def __init__(
            self, 
            backbone_config, 
            head_config, 
        ):
        super(EmbeddigsNet, self).__init__()
        backbone, backbone_out_feats = get_backbone(backbone_config)
        self.head = get_head(head_config, backbone_out_feats)
        self.backbone = backbone


    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


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
        backbone_config, 
        head_config
    ):

    model = EmbeddigsNet(
        backbone_config=backbone_config, 
        head_config=head_config
    )
    return model