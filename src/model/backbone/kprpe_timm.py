from types import MethodType

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.kp_rpe import KeypointRelativePositionBias


def _kprpe_attention_forward(self, x):
    batch_size, n_tokens, channels = x.shape
    qkv = self.qkv(x).reshape(
        batch_size,
        n_tokens,
        3,
        self.num_heads,
        channels // self.num_heads,
    )
    qkv = qkv.permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    if hasattr(self, "q_norm"):
        q = self.q_norm(q)
    if hasattr(self, "k_norm"):
        k = self.k_norm(k)

    scale = getattr(self, "scale", q.shape[-1] ** -0.5)
    attn = (q * scale) @ k.transpose(-2, -1)
    bias = getattr(self, "kprpe_bias", None)
    if bias is not None:
        if bias.shape[-2:] != attn.shape[-2:]:
            raise ValueError(
                "KP-RPE bias token shape "
                f"{tuple(bias.shape[-2:])} does not match attention "
                f"{tuple(attn.shape[-2:])}"
            )
        attn = attn + bias.to(dtype=attn.dtype)

    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    x = attn @ v
    x = x.transpose(1, 2).reshape(batch_size, n_tokens, channels)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def _iter_vit_attention_modules(model):
    for block in getattr(model, "blocks", []):
        attn = getattr(block, "attn", None)
        if attn is not None and all(
            hasattr(attn, attr)
            for attr in ("qkv", "num_heads", "attn_drop", "proj", "proj_drop")
        ):
            yield attn


def install_kprpe_attention(model):
    attention_modules = list(_iter_vit_attention_modules(model))
    if not attention_modules:
        raise ValueError("KP-RPE requires a timm ViT-style model with block.attn.qkv")

    for attn in attention_modules:
        if hasattr(attn, "fused_attn"):
            attn.fused_attn = False
        attn.forward = MethodType(_kprpe_attention_forward, attn)
        attn.kprpe_bias = None

    return attention_modules


class KPRPETimmBackbone(nn.Module):
    supports_keypoints = True

    def __init__(
        self,
        model_type,
        pretrained=True,
        grid_size=(14, 14),
        num_keypoints=5,
        hidden_dim=64,
        aggregate="mean",
        scriptable=False,
        grad_checkpointing=False,
    ):
        super(KPRPETimmBackbone, self).__init__()
        self.backbone = timm.create_model(
            model_type,
            pretrained=pretrained,
            scriptable=scriptable,
            num_classes=0,
        )
        if grad_checkpointing:
            set_grad_checkpointing = getattr(
                self.backbone, "set_grad_checkpointing", None
            )
            if set_grad_checkpointing is None:
                raise ValueError(
                    f"Backbone {model_type} does not support grad_checkpointing"
                )
            set_grad_checkpointing(True)
        self.num_features = getattr(self.backbone, "num_features", None)
        if self.num_features is None:
            self.num_features = getattr(self.backbone, "embed_dim")

        attention_modules = install_kprpe_attention(self.backbone)
        num_heads = attention_modules[0].num_heads
        include_cls_token = bool(getattr(self.backbone, "cls_token", None) is not None)
        self.kprpe = KeypointRelativePositionBias(
            num_heads=num_heads,
            grid_size=grid_size,
            num_keypoints=num_keypoints,
            hidden_dim=hidden_dim,
            include_cls_token=include_cls_token,
            aggregate=aggregate,
        )
        self.attention_modules = attention_modules

    def _set_attention_bias(self, bias):
        for attn in self.attention_modules:
            attn.kprpe_bias = bias

    def forward(self, x, keypoints=None):
        if keypoints is None:
            return self.backbone(x)

        bias = self.kprpe(keypoints, pairwise=True)
        self._set_attention_bias(bias)
        try:
            return self.backbone(x)
        finally:
            self._set_attention_bias(None)


def load_kprpe_timm_model(backbone_config):
    return KPRPETimmBackbone(
        model_type=backbone_config.model_type,
        pretrained=backbone_config.pretrained,
        grid_size=getattr(backbone_config.kprpe, "grid_size", (14, 14)),
        num_keypoints=backbone_config.kprpe.num_keypoints,
        hidden_dim=backbone_config.kprpe.hidden_dim,
        aggregate=getattr(backbone_config.kprpe, "aggregate", "mean"),
        scriptable=getattr(backbone_config, "scriptable", False),
        grad_checkpointing=getattr(backbone_config, "grad_checkpointing", False),
    )
