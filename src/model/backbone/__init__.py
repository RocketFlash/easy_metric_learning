import torch.nn as nn

import timm

from .unicom import load_model_unicom
from .iresnet import load_iresnet
from .edgeface import EDGEFACE_VARIANTS, load_edgeface
from .kprpe_timm import load_kprpe_timm_model
from .radio import load_radio
from .mobilefacenet import load_mobilefacenet

IRESNET_TYPES = {"iresnet50", "iresnet100", "iresnet200"}
MOBILEFACENET_TYPES = {"mobilefacenet"}


def disable_info(*args, **kwargs):
    pass


def load_openclip_model(model_type, pretrained_on):
    import open_clip

    info_fnc = open_clip.create_model_and_transforms.__globals__["logging"].info
    open_clip.create_model_and_transforms.__globals__["logging"].info = disable_info

    clip_model, _, _ = open_clip.create_model_and_transforms(
        model_type, pretrained=pretrained_on
    )

    # open_clip.create_model_and_transforms.__globals__['logging'].info = info_fnc

    backbone = clip_model.visual
    return backbone


def _config_value(config, key, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _maybe_set_grad_checkpointing(backbone, backbone_config):
    grad_checkpointing = _config_value(backbone_config, "grad_checkpointing", False)
    if not grad_checkpointing:
        return

    set_grad_checkpointing = getattr(backbone, "set_grad_checkpointing", None)
    if set_grad_checkpointing is None:
        backbone_type = _config_value(backbone_config, "type", "<unknown>")
        raise ValueError(
            f"Backbone {backbone_type} does not support grad_checkpointing"
        )
    set_grad_checkpointing(True)


def load_timm_model(backbone_config):
    backbone = timm.create_model(
        backbone_config.type,
        pretrained=backbone_config.pretrained,
        scriptable=_config_value(backbone_config, "scriptable", False),
        num_classes=0,
    )
    _maybe_set_grad_checkpointing(backbone, backbone_config)
    return backbone


def load_openclip_model_from_config(backbone_config):
    return load_openclip_model(
        backbone_config.model_type, pretrained_on=backbone_config.pretrained_on
    )


def load_unicom_model_from_config(backbone_config):
    return load_model_unicom(backbone_config.type)


def get_openclip_out_features(backbone):
    if hasattr(backbone, "output_dim"):
        return backbone.output_dim
    if hasattr(backbone, "head") and hasattr(backbone.head, "proj"):
        backbone_out_feats = backbone.head.proj.in_features
        backbone.head.proj = nn.Identity()
        return backbone_out_feats
    if hasattr(backbone, "head") and hasattr(backbone.head, "in_features"):
        backbone_out_feats = backbone.head.in_features
        backbone.head = nn.Identity()
        return backbone_out_feats

    raise ValueError("Could not infer output features for OpenCLIP visual backbone")


def get_timm_out_features(backbone, backbone_type):
    if hasattr(backbone, "num_features"):
        return backbone.num_features
    if hasattr(backbone, "embed_dim"):
        return backbone.embed_dim
    if hasattr(backbone, "head") and hasattr(backbone.head, "fc"):
        backbone_out_feats = backbone.head.fc.in_features
        backbone.head.fc = nn.Identity()
        return backbone_out_feats
    if hasattr(backbone, "head") and hasattr(backbone.head, "in_features"):
        backbone_out_feats = backbone.head.in_features
        backbone.head = nn.Identity()
        return backbone_out_feats
    if hasattr(backbone, "classifier"):
        backbone_out_feats = backbone.classifier.in_features
        backbone.classifier = nn.Identity()
        return backbone_out_feats
    if hasattr(backbone, "fc"):
        backbone_out_feats = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return backbone_out_feats

    raise ValueError(
        f"Could not infer output features for timm backbone {backbone_type}"
    )


def _num_features(backbone, backbone_config):
    return backbone.num_features


def _unicom_out_features(backbone, backbone_config):
    return backbone.embedding_size


def _openclip_out_features(backbone, backbone_config):
    return get_openclip_out_features(backbone)


def _timm_out_features(backbone, backbone_config):
    return get_timm_out_features(backbone, backbone_config.type)


BACKBONE_REGISTRY = {
    "timm": (load_timm_model, _timm_out_features),
    "openclip": (load_openclip_model_from_config, _openclip_out_features),
    "unicom": (load_unicom_model_from_config, _unicom_out_features),
    "iresnet": (load_iresnet, _num_features),
    "edgeface": (load_edgeface, _num_features),
    "kprpe_timm": (load_kprpe_timm_model, _num_features),
    "radio": (load_radio, _num_features),
    "mobilefacenet": (load_mobilefacenet, _num_features),
}


def infer_backbone_family(backbone_config):
    family = _config_value(backbone_config, "family", None)
    if family:
        return family

    backbone_type = backbone_config.type
    if backbone_type.startswith("openclip-"):
        return "openclip"
    if backbone_type.startswith("unicom_"):
        return "unicom"
    if backbone_type in IRESNET_TYPES:
        return "iresnet"
    if backbone_type in MOBILEFACENET_TYPES:
        return "mobilefacenet"
    if backbone_type in EDGEFACE_VARIANTS:
        return "edgeface"
    if backbone_type.startswith("kprpe_"):
        return "kprpe_timm"
    if backbone_type.startswith(("radio_", "c_radio_", "am_radio_")):
        return "radio"
    return "timm"


def get_backbone(backbone_config):
    family = infer_backbone_family(backbone_config)
    if family not in BACKBONE_REGISTRY:
        valid_families = ", ".join(sorted(BACKBONE_REGISTRY))
        raise ValueError(
            f"Unknown backbone family '{family}'. Expected one of: {valid_families}"
        )

    load_backbone, get_out_features = BACKBONE_REGISTRY[family]
    backbone = load_backbone(backbone_config)
    backbone_out_feats = get_out_features(backbone, backbone_config)

    if backbone_config.freeze:
        for param in backbone.parameters():
            param.requires_grad = False

    return backbone, backbone_out_feats
