from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from src.model.backbone import get_backbone, get_openclip_out_features
from src.model.backbone.iresnet import iresnet50
from src.model.backbone.mobilefacenet import MobileFaceNet


class FakeTimmBackbone(torch.nn.Module):
    def __init__(self):
        super(FakeTimmBackbone, self).__init__()
        self.num_features = 768
        self.param = torch.nn.Parameter(torch.ones(1))
        self.grad_checkpointing = False

    def set_grad_checkpointing(self, enabled=True):
        self.grad_checkpointing = enabled

    def forward(self, x):
        return torch.ones(x.size(0), self.num_features, device=x.device)


class FakeEdgeNextBackbone(torch.nn.Module):
    def __init__(self):
        super(FakeEdgeNextBackbone, self).__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.ReLU(),
        )
        self.head = torch.nn.Linear(4, 4)
        self.num_features = 4

    def reset_classifier(self, num_classes):
        self.num_features = num_classes
        self.head = torch.nn.Linear(4, num_classes)

    def forward(self, x):
        x = torch.ones(x.size(0), 4, device=x.device)
        return self.head(self.block(x))


class FakeOpenClipBackbone(torch.nn.Module):
    output_dim = 512


class FakeRadioModel(torch.nn.Module):
    def __init__(self):
        super(FakeRadioModel, self).__init__()
        self.param = torch.nn.Parameter(torch.ones(1))

    def get_nearest_supported_resolution(self, height, width):
        return height + 1, width + 1

    def forward(self, x):
        return x.mean(dim=(2, 3)), x


class FakeAttention(torch.nn.Module):
    def __init__(self):
        super(FakeAttention, self).__init__()
        self.num_heads = 2
        self.qkv = torch.nn.Linear(4, 12)
        self.attn_drop = torch.nn.Dropout(0.0)
        self.proj = torch.nn.Linear(4, 4)
        self.proj_drop = torch.nn.Dropout(0.0)
        self.scale = 2**-0.5


class FakeViTBlock(torch.nn.Module):
    def __init__(self):
        super(FakeViTBlock, self).__init__()
        self.attn = FakeAttention()

    def forward(self, x):
        return self.attn(x)


class FakeViTBackbone(torch.nn.Module):
    def __init__(self):
        super(FakeViTBackbone, self).__init__()
        self.num_features = 4
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, 4))
        self.blocks = torch.nn.ModuleList([FakeViTBlock()])
        self.grad_checkpointing = False

    def set_grad_checkpointing(self, enabled=True):
        self.grad_checkpointing = enabled

    def forward(self, x):
        batch_size = x.size(0)
        tokens = torch.ones(batch_size, 5, 4, device=x.device)
        tokens = self.blocks[0](tokens)
        return tokens.mean(dim=1)


def test_timm_backbone_is_created_as_feature_extractor(monkeypatch):
    captured = {}

    def fake_create_model(*args, **kwargs):
        captured["model_name"] = args[0]
        captured.update(kwargs)
        return FakeTimmBackbone()

    monkeypatch.setattr("src.model.backbone.timm.create_model", fake_create_model)

    backbone, out_features = get_backbone(
        SimpleNamespace(
            type="vit_base_patch14_dinov2.lvd142m",
            family="timm",
            pretrained=True,
            scriptable=True,
            grad_checkpointing=True,
            freeze=True,
        )
    )

    assert captured["num_classes"] == 0
    assert captured["scriptable"] is True
    assert captured["model_name"] == "vit_base_patch14_dinov2.lvd142m"
    assert out_features == 768
    assert backbone.grad_checkpointing is True
    assert all(not param.requires_grad for param in backbone.parameters())


def test_timm_backbone_defaults_to_non_scriptable(monkeypatch):
    captured = {}

    def fake_create_model(*args, **kwargs):
        captured.update(kwargs)
        return FakeTimmBackbone()

    monkeypatch.setattr("src.model.backbone.timm.create_model", fake_create_model)

    get_backbone(
        SimpleNamespace(
            type="vit_base_patch16_siglip_224.v2_webli",
            family="timm",
            pretrained=True,
            freeze=False,
        )
    )

    assert captured["scriptable"] is False


def test_unknown_backbone_family_raises():
    with pytest.raises(ValueError, match="Unknown backbone family"):
        get_backbone(
            SimpleNamespace(
                type="vit_base_patch16_224",
                family="missing",
                pretrained=False,
                freeze=False,
            )
        )


def test_iresnet50_outputs_configured_feature_size():
    model = iresnet50(num_features=32)
    model.eval()

    with torch.no_grad():
        output = model(torch.randn(2, 3, 112, 112))

    assert output.shape == (2, 32)


def test_mobilefacenet_outputs_configured_feature_size():
    model = MobileFaceNet(num_features=32)
    model.eval()

    with torch.no_grad():
        output = model(torch.randn(2, 3, 112, 112))

    assert output.shape == (2, 32)


def test_iresnet_backbone_factory_reports_num_features():
    backbone, out_features = get_backbone(
        SimpleNamespace(
            type="iresnet50",
            family="iresnet",
            pretrained=False,
            freeze=True,
            dropout=0.0,
            num_features=64,
        )
    )

    assert out_features == 64
    assert all(not param.requires_grad for param in backbone.parameters())


def test_mobilefacenet_backbone_factory_reports_num_features():
    backbone, out_features = get_backbone(
        SimpleNamespace(
            type="mobilefacenet",
            family="mobilefacenet",
            pretrained=False,
            checkpoint_path=None,
            freeze=True,
            dropout=0.0,
            expansion=2,
            num_features=64,
        )
    )

    assert out_features == 64
    assert all(not param.requires_grad for param in backbone.parameters())


def test_edgeface_backbone_uses_edgenext_lowrank_variant(monkeypatch):
    captured = {}

    def fake_create_model(model_name, **kwargs):
        captured["model_name"] = model_name
        captured.update(kwargs)
        return FakeEdgeNextBackbone()

    monkeypatch.setattr(
        "src.model.backbone.edgeface.timm.create_model", fake_create_model
    )

    backbone, out_features = get_backbone(
        SimpleNamespace(
            type="edgeface_xs_gamma_06",
            family="edgeface",
            pretrained=False,
            checkpoint_path=None,
            freeze=False,
            num_features=16,
        )
    )

    assert captured["model_name"] == "edgenext_x_small"
    assert out_features == 16
    assert backbone.model.block[0].__class__.__name__ == "LowRankLinear"
    assert isinstance(backbone.model.head, torch.nn.Linear)


def test_kprpe_timm_backbone_accepts_keypoints(monkeypatch):
    captured = {}

    def fake_create_model(*args, **kwargs):
        captured["model_name"] = args[0]
        captured.update(kwargs)
        return FakeViTBackbone()

    monkeypatch.setattr(
        "src.model.backbone.kprpe_timm.timm.create_model",
        fake_create_model,
    )

    backbone, out_features = get_backbone(
        SimpleNamespace(
            type="kprpe_vit_base_patch16_224",
            family="kprpe_timm",
            model_type="vit_base_patch16_224",
            pretrained=False,
            scriptable=True,
            grad_checkpointing=True,
            freeze=False,
            kprpe=SimpleNamespace(
                grid_size=(2, 2),
                num_keypoints=5,
                hidden_dim=8,
                aggregate="mean",
            ),
        )
    )

    output = backbone(torch.randn(2, 3, 16, 16), keypoints=torch.rand(2, 5, 2))

    assert captured["model_name"] == "vit_base_patch16_224"
    assert captured["scriptable"] is True
    assert backbone.backbone.grad_checkpointing is True
    assert out_features == 4
    assert output.shape == (2, 4)


def test_radio_backbone_uses_torchhub_and_summary_output(monkeypatch):
    captured = {}

    def fake_torchhub_load(repo, entrypoint, **kwargs):
        captured["repo"] = repo
        captured["entrypoint"] = entrypoint
        captured.update(kwargs)
        return FakeRadioModel()

    monkeypatch.setattr("src.model.backbone.radio.torch.hub.load", fake_torchhub_load)

    backbone, out_features = get_backbone(
        SimpleNamespace(
            type="am_radio_v2_5_b",
            family="radio",
            repo="NVlabs/RADIO",
            entrypoint="radio_model",
            version="radio_v2.5-b",
            pretrained=True,
            progress=False,
            skip_validation=True,
            resize_to_supported_resolution=True,
            output_key="backbone",
            freeze=False,
            num_features=3,
        )
    )

    output = backbone(torch.ones(2, 3, 8, 8))

    assert captured["repo"] == "NVlabs/RADIO"
    assert captured["entrypoint"] == "radio_model"
    assert captured["version"] == "radio_v2.5-b"
    assert out_features == 3
    assert output.shape == (2, 3)


def test_openclip_feature_detection_uses_output_dim():
    assert get_openclip_out_features(FakeOpenClipBackbone()) == 512
