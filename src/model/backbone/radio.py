import torch
import torch.nn as nn
import torch.nn.functional as F


class RadioBackbone(nn.Module):
    def __init__(
        self,
        repo="NVlabs/RADIO",
        entrypoint="radio_model",
        version="radio_v2.5-b",
        num_features=768,
        progress=True,
        skip_validation=True,
        resize_to_supported_resolution=True,
        output_key="backbone",
    ):
        super(RadioBackbone, self).__init__()
        self.num_features = int(num_features)
        self.version = version
        self.resize_to_supported_resolution = resize_to_supported_resolution
        self.output_key = output_key
        self.model = torch.hub.load(
            repo,
            entrypoint,
            version=version,
            progress=progress,
            skip_validation=skip_validation,
        )

    def _maybe_resize(self, x):
        if not self.resize_to_supported_resolution:
            return x
        nearest_resolution = getattr(
            self.model, "get_nearest_supported_resolution", None
        )
        if nearest_resolution is None:
            return x
        height, width = nearest_resolution(*x.shape[-2:])
        if (height, width) == tuple(x.shape[-2:]):
            return x
        return F.interpolate(
            x,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

    def _extract_summary(self, output):
        if isinstance(output, dict):
            output = output[self.output_key]
        if hasattr(output, "summary"):
            return output.summary
        if isinstance(output, (list, tuple)):
            return output[0]
        return output

    def forward(self, x):
        x = self._maybe_resize(x)
        return self._extract_summary(self.model(x))


def load_radio(backbone_config):
    if not getattr(backbone_config, "pretrained", True):
        raise ValueError("RADIO backbones are loaded from pretrained torch.hub weights")

    return RadioBackbone(
        repo=getattr(backbone_config, "repo", "NVlabs/RADIO"),
        entrypoint=getattr(backbone_config, "entrypoint", "radio_model"),
        version=getattr(backbone_config, "version", "radio_v2.5-b"),
        num_features=getattr(backbone_config, "num_features", 768),
        progress=getattr(backbone_config, "progress", True),
        skip_validation=getattr(backbone_config, "skip_validation", True),
        resize_to_supported_resolution=getattr(
            backbone_config,
            "resize_to_supported_resolution",
            True,
        ),
        output_key=getattr(backbone_config, "output_key", "backbone"),
    )
