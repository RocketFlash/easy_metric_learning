import timm
import torch.nn as nn
import functools
from .unicom import load_model_unicom


def disable_info(*args, **kwargs):
    pass


def load_openclip_model(model_type, pretrained_on):
    import open_clip
    
    info_fnc = open_clip.create_model_and_transforms.__globals__['logging'].info
    open_clip.create_model_and_transforms.__globals__['logging'].info = disable_info
    
    clip_model, _, _ = open_clip.create_model_and_transforms(
        model_type, 
        pretrained=pretrained_on
    )

    # open_clip.create_model_and_transforms.__globals__['logging'].info = info_fnc 

    backbone = clip_model.visual
    return backbone


def get_backbone(backbone_config):
    backbone_type = backbone_config.type

    if 'openclip' in backbone_type:
        backbone = load_openclip_model(
            backbone_config.model_type, 
            pretrained_on=backbone_config.pretrained_on
        )
    elif 'unicom' in backbone_type:
        backbone = load_model_unicom(backbone_type)
    else:
        backbone = timm.create_model(backbone_type, 
                                     pretrained=backbone_config.pretrained, 
                                     scriptable=True)  
    
    if 'openclip' in backbone_type:
        if 'ViT' in backbone_type:
            backbone_out_feats = backbone.output_dim
        else:
            backbone_out_feats = backbone.head.proj.in_features
            backbone.head.proj = nn.Identity()
    elif 'unicom' in backbone_type:
        backbone_out_feats = backbone.embedding_size
    elif 'maxvit' in backbone_type or\
       'convnext' in backbone_type or\
       'coatnet' in backbone_type:
        backbone_out_feats = backbone.head.fc.in_features
        backbone.head.fc = nn.Identity()
    elif 'swin' in backbone_type:
        backbone_out_feats = backbone.head.in_features
        backbone.head = nn.Identity()
    elif 'vit' in backbone_type:
        backbone_out_feats = backbone.embed_dim
    elif 'efficientnet' in backbone_type:
        backbone_out_feats = backbone.classifier.in_features
        backbone.classifier = nn.Identity()
    else:
        backbone_out_feats = backbone.fc.in_features
        backbone.fc = nn.Identity()


    if backbone_config.freeze:
        for param in backbone.parameters():
            param.requires_grad = False
        
    return backbone, backbone_out_feats