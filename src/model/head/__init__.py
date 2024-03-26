from .base import BaseHead
import torch.nn as nn


def get_head(
        head_config, 
        backbone_out_feats
    ):
    head_type = head_config.type

    if head_type == 'base':
        return BaseHead(
            backbone_out_feats=backbone_out_feats,
            embeddings_size=head_config.embeddings_size,
            dropout_p=head_config.dropout_p)
    else:
        return nn.Identity()
    