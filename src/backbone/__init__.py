from .iresnet import iresnet18, iresnet34, iresnet50, iresnet100, iresnet200

def get_backbone(name, **kwargs):
    # resnet
    if name == "iresnet18":
        return iresnet18(False, **kwargs)
    elif name == "iresnet34":
        return iresnet34(False, **kwargs)
    elif name == "iresnet50":
        return iresnet50(False, **kwargs)
    elif name == "iresnet100":
        return iresnet100(False, **kwargs)
    elif name == "iresnet200":
        return iresnet200(False, **kwargs)
    else:
        raise ValueError()