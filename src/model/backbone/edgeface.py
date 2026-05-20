import torch
import torch.nn as nn
import timm


class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True):
        super(LowRankLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.linear1 = nn.Linear(in_features, rank, bias=False)
        self.linear2 = nn.Linear(rank, out_features, bias=bias)

    def forward(self, x):
        return self.linear2(self.linear1(x))


def replace_linear_with_lowrank(module, rank_ratio=0.2, skip_names=("head",)):
    for name, child in module.named_children():
        if any(skip in name for skip in skip_names):
            continue
        if isinstance(child, nn.Linear):
            rank = max(2, int(min(child.in_features, child.out_features) * rank_ratio))
            setattr(
                module,
                name,
                LowRankLinear(
                    child.in_features,
                    child.out_features,
                    rank=rank,
                    bias=child.bias is not None,
                ),
            )
        else:
            replace_linear_with_lowrank(child, rank_ratio, skip_names=skip_names)
    return module


class EdgeFaceBackbone(nn.Module):
    def __init__(self, model_name, num_features=512, rank_ratio=None):
        super(EdgeFaceBackbone, self).__init__()
        self.num_features = num_features
        self.model_name = model_name
        self.model = timm.create_model(model_name, pretrained=False)
        self.model.reset_classifier(num_features)

        if rank_ratio is not None:
            self.model = replace_linear_with_lowrank(
                self.model,
                rank_ratio=rank_ratio,
            )

    def forward(self, x):
        return self.model(x)


EDGEFACE_VARIANTS = {
    "edgeface_xs_gamma_06": ("edgenext_x_small", 0.6),
    "edgeface_xxs": ("edgenext_xx_small", None),
    "edgeface_base": ("edgenext_base", None),
    "edgeface_s_gamma_05": ("edgenext_small", 0.5),
}


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "module", "net"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def load_edgeface(backbone_config):
    try:
        model_name, rank_ratio = EDGEFACE_VARIANTS[backbone_config.type]
    except KeyError as exc:
        raise ValueError(f"Unknown EdgeFace backbone: {backbone_config.type}") from exc

    model = EdgeFaceBackbone(
        model_name=model_name,
        num_features=getattr(backbone_config, "num_features", 512),
        rank_ratio=rank_ratio,
    )

    checkpoint_path = getattr(backbone_config, "checkpoint_path", None)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(_extract_state_dict(checkpoint), strict=True)
    elif getattr(backbone_config, "pretrained", False):
        raise ValueError(
            "Pretrained EdgeFace weights are not bundled with this repo. "
            "Set checkpoint_path to a downloaded EdgeFace checkpoint."
        )

    return model
