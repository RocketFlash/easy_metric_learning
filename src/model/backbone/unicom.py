import hashlib
import os
import urllib
import warnings

import torch
from typing import List
from tqdm import tqdm

import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from torch.utils.checkpoint import checkpoint

# Code from Unicom repository https://github.com/deepglint/unicom/blob/main/unicom/model.py

_MODELS = {
    "unicom_vit_b32": "https://github.com/deepglint/unicom/releases/download/b32/FP16-ViT-B-32.pt",
    "unicom_vit_b16": "https://github.com/deepglint/unicom/releases/download/b16/FP16-ViT-B-16.pt",
    "unicom_vit_l14": "https://github.com/deepglint/unicom/releases/download/l14/FP16-ViT-L-14.pt",
    "unicom_vit_l14_336": "https://github.com/deepglint/unicom/releases/download/l14_336px/FP16-ViT-L-14-336px.pt",
}

_SHA256 = {
    "FP16-ViT-B-32.pt": "f9d5696a9b58dbbbefee2d31615ca59084f2895a0fdd2ca4c235e0f9b2793f7a",
    "FP16-ViT-B-16.pt": "c04f324f7c3b4435667236ec6c0eca1cd62f9d64fbfc2d06f8e8e60e6497edef",
    "FP16-ViT-L-14.pt": "ff3ab62ff782876460099e6e0ee17b73a7c01109de2fffd595f16f4129404bbd",
    "FP16-ViT-L-14-336px.pt": "3916ab5aed3b522fc90345be8b4457fe5dad60801ad2af5a6871c0c096e8d7ea",
}


def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


def rm_module_from_state_dict(state_dict: dict) -> dict:
    result = {}
    for k, value in state_dict.items():

        if "module." in k:
            k_removed = k.split("module.")[-1]
            result[k_removed] = value
        else:
            result[k] = value
    return result


# copy from https://github.com/openai/CLIP/blob/main/clip/clip.py#L43
def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = _SHA256[filename]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(
            f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(
                f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(
            "Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


# copy from https://github.com/openai/CLIP/blob/main/clip/clip.py#L94
def load_model_unicom(name: str, download_root: str = None):
    if name in _MODELS:
        model_path = _download(
            _MODELS[name], download_root or os.path.expanduser("~/.cache/unicom"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(
            f"Model {name} not found; available models = {available_models()}")
    with open(model_path, 'rb') as opened_file:
        state_dict = torch.load(opened_file)

    model = load_model(name)
    state_dict_fp32 = {}
    for k, v in state_dict.items():
        state_dict_fp32[k] = v.float()

    model.load_state_dict(state_dict)
    return model


class VisionTransformer(nn.Module):
    def __init__(self, input_size=224, patch_size=32, in_channels=3, dim=768, embedding_size=768,
                 depth=12, num_heads=12, mlp_ratio=4, drop_path_rate=0.0, using_checkpoint=True):
        super().__init__()
        self.dim = dim
        self.patch_embed = PatchEmbedding(
            input_size, patch_size, in_channels, dim,)
        self.pos_embed = nn.Parameter(torch.zeros(
            1, self.patch_embed.num_patches, dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList(
            [
                Block(dim, num_heads, mlp_ratio, dpr[i], self.patch_embed.num_patches, using_checkpoint) for i in range(depth)
            ])
        self.norm = nn.LayerNorm(dim)
        self.embedding_size = embedding_size

        self.feature = nn.Sequential(
            nn.Linear(dim * self.patch_embed.num_patches, dim, False),
            nn.BatchNorm1d(dim, eps=2e-5),
            nn.Linear(dim, embedding_size, False),
            nn.BatchNorm1d(embedding_size, eps=2e-5))

        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)
        self.extra_gflops = 0.0
        for _block in self.blocks:
            self.extra_gflops += _block.extra_gflops

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        for func in self.blocks:
            x = func(x)
        x = self.norm(x.float())
        return torch.reshape(x, (B, self.patch_embed.num_patches * self.dim))

    def forward(self, x):
        x = self.forward_features(x)
        x = self.feature(x)
        return x


class Mlp(nn.Module):
    def __init__(self, dim, dim_hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim_hidden)
        self.act = nn.ReLU6()
        self.fc2 = nn.Linear(dim_hidden, dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        with torch.cuda.amp.autocast(True):
            B, L, D = x.shape
            qkv = self.qkv(x).reshape(B, L, 3, self.num_heads,
                                      D // self.num_heads).permute(2, 0, 3, 1, 4)
        with torch.cuda.amp.autocast(False):
            q, k, v = qkv[0].float(), qkv[1].float(), qkv[2].float()
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            x = (attn @ v).transpose(1, 2).reshape(B, L, D)
        with torch.cuda.amp.autocast(True):
            x = self.proj(x)
        return x


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4, drop_path: float = 0.0, patch_n: int = 32, using_checkpoint=False):
        super().__init__()
        self.using_checkpoint = using_checkpoint
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        if drop_path > 0:
            self.drop_path = DropPath(drop_path)
        else:
            self.drop_path = nn.Identity()
        self.mlp = Mlp(dim, dim * mlp_ratio)
        self.extra_gflops = (num_heads * patch_n * (dim // num_heads) * patch_n * 2) / (1000**3)

    def forward_impl(self, x):
        with torch.cuda.amp.autocast(True):
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, x):
        if self.using_checkpoint:
            return checkpoint(self.forward_impl, x, use_reentrant=False)
        else:
            return self.forward_impl(x)


class PatchEmbedding(nn.Module):
    def __init__(self, input_size=224, patch_size=32, in_channels: int = 3, dim: int = 768):
        super().__init__()
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        H = input_size[0] // patch_size[0]
        W = input_size[1] // patch_size[1]
        self.num_patches = H * W
        self.proj = nn.Conv2d(
            in_channels, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


def build_model(name="ViT-L/14@336px"):
    if name == "unicom_vit_b32":
        model = VisionTransformer(
            input_size=224, patch_size=32, in_channels=3, dim=768, embedding_size=512,
            depth=12, num_heads=12, drop_path_rate=0.1, using_checkpoint=True)
    elif name == "unicom_vit_b16":
        model = VisionTransformer(
            input_size=224, patch_size=16, in_channels=3, dim=768, embedding_size=768,
            depth=12, num_heads=12, drop_path_rate=0.1, using_checkpoint=True)
    elif name == "unicom_vit_l14":
        model = VisionTransformer(
            input_size=224, patch_size=14, in_channels=3, dim=1024, embedding_size=768,
            depth=24, num_heads=16, drop_path_rate=0.1, using_checkpoint=True)
    elif name == "unicom_vit_l14_336":
        model = VisionTransformer(
            input_size=336, patch_size=14, in_channels=3, dim=1024, embedding_size=768,
            depth=24, num_heads=16, drop_path_rate=0.1, using_checkpoint=True)
    return model


def load_model(name="unicom_vit_b32"):
    if name == "unicom_vit_b32":
        return build_model(name)
    elif name == "unicom_vit_b16":
        return build_model(name)
    elif name == "unicom_vit_l14":
        return build_model(name)
    elif name == "unicom_vit_l14_336":
        return build_model(name)
    else:
        raise