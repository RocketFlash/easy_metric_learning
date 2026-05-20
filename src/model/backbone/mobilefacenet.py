import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, groups=1
    ):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )

    def forward(self, x):
        return self.block(x)


class LinearBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, groups=1
    ):
        super(LinearBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.block(x)


class DepthWiseBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, residual=False, stride=1, expansion=2
    ):
        super(DepthWiseBlock, self).__init__()
        hidden_channels = int(in_channels * expansion)
        self.residual = residual and stride == 1 and in_channels == out_channels
        self.conv = nn.Sequential(
            ConvBlock(in_channels, hidden_channels, 1, 1, 0),
            ConvBlock(
                hidden_channels,
                hidden_channels,
                3,
                stride,
                1,
                groups=hidden_channels,
            ),
            LinearBlock(hidden_channels, out_channels, 1, 1, 0),
        )

    def forward(self, x):
        out = self.conv(x)
        if self.residual:
            out = out + x
        return out


def _make_stage(in_channels, out_channels, num_blocks, stride, expansion):
    layers = [
        DepthWiseBlock(
            in_channels,
            out_channels,
            residual=False,
            stride=stride,
            expansion=expansion,
        )
    ]
    for _ in range(1, int(num_blocks)):
        layers.append(
            DepthWiseBlock(
                out_channels,
                out_channels,
                residual=True,
                stride=1,
                expansion=expansion,
            )
        )
    return nn.Sequential(*layers)


class MobileFaceNet(nn.Module):
    def __init__(self, num_features=512, dropout=0.0, expansion=2):
        super(MobileFaceNet, self).__init__()
        self.num_features = int(num_features)
        self.conv1 = ConvBlock(3, 64, 3, 2, 1)
        self.conv2_dw = ConvBlock(64, 64, 3, 1, 1, groups=64)
        self.stage1 = _make_stage(64, 64, num_blocks=5, stride=2, expansion=expansion)
        self.stage2 = _make_stage(64, 128, num_blocks=1, stride=2, expansion=expansion)
        self.stage3 = _make_stage(128, 128, num_blocks=6, stride=1, expansion=expansion)
        self.stage4 = _make_stage(128, 128, num_blocks=1, stride=2, expansion=expansion)
        self.stage5 = _make_stage(128, 128, num_blocks=2, stride=1, expansion=expansion)
        self.conv_sep = ConvBlock(128, 512, 1, 1, 0)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(512, self.num_features)
        self.features = nn.BatchNorm1d(self.num_features)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False
        self._init_params()

    def _init_params(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_dw(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.conv_sep(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.features(x)
        return x


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "module", "net"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def load_mobilefacenet(backbone_config):
    model = MobileFaceNet(
        num_features=getattr(backbone_config, "num_features", 512),
        dropout=getattr(backbone_config, "dropout", 0.0),
        expansion=getattr(backbone_config, "expansion", 2),
    )

    checkpoint_path = getattr(backbone_config, "checkpoint_path", None)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(_extract_state_dict(checkpoint), strict=True)
    elif getattr(backbone_config, "pretrained", False):
        raise ValueError(
            "Pretrained MobileFaceNet weights are not bundled with this repo. "
            "Set checkpoint_path to a downloaded MobileFaceNet checkpoint."
        )

    return model
