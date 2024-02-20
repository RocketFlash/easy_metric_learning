import timm
import torch.nn as nn


def get_backbone(backbone_config):
    backbone_type = backbone_config.type

    if 'openclip' in backbone_type:
        import open_clip
        clip_model, _, _ = open_clip.create_model_and_transforms(backbone_config.model_type, 
                                                                 pretrained=backbone_config.pretrained_on)
        backbone = clip_model.visual
        
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