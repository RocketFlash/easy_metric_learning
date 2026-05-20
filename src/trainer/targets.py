import torch


def split_targets(targets):
    if isinstance(targets, dict):
        label = targets.get("label", targets.get("labels"))
        if label is None:
            raise ValueError("Target dictionaries must include 'label' or 'labels'")
        extras = {
            key: value
            for key, value in targets.items()
            if key not in {"label", "labels"}
        }
        return label, extras
    return targets, {}


def move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def get_keypoints(target_extras):
    return target_extras.get("keypoints") if target_extras else None
