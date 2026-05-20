import torch
import torch.nn as nn


def build_patch_grid(grid_size, device=None, dtype=None):
    if isinstance(grid_size, int):
        grid_h = grid_w = grid_size
    else:
        grid_h, grid_w = grid_size

    y = torch.linspace(0.0, 1.0, grid_h, device=device, dtype=dtype)
    x = torch.linspace(0.0, 1.0, grid_w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([xx, yy], dim=-1).reshape(-1, 2)


class KeypointRelativePositionBias(nn.Module):
    """Keypoint-conditioned relative attention bias for ViT patch tokens."""

    def __init__(
        self,
        num_heads,
        grid_size,
        num_keypoints,
        hidden_dim=64,
        include_cls_token=False,
        aggregate="mean",
    ):
        super(KeypointRelativePositionBias, self).__init__()
        if aggregate not in {"mean", "max", "sum"}:
            raise ValueError("aggregate must be one of: mean, max, sum")

        self.num_heads = num_heads
        self.grid_size = grid_size
        self.num_keypoints = num_keypoints
        self.include_cls_token = include_cls_token
        self.aggregate = aggregate
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_heads),
        )

    def _aggregate(self, token_keypoint_bias):
        if self.aggregate == "mean":
            return token_keypoint_bias.mean(dim=2)
        if self.aggregate == "sum":
            return token_keypoint_bias.sum(dim=2)
        return token_keypoint_bias.max(dim=2).values

    def forward(self, keypoints, pairwise=True):
        if keypoints.ndim != 3 or keypoints.shape[1:] != (self.num_keypoints, 2):
            raise ValueError(
                "keypoints must have shape "
                f"[B, {self.num_keypoints}, 2], got {tuple(keypoints.shape)}"
            )

        token_grid = build_patch_grid(
            self.grid_size,
            device=keypoints.device,
            dtype=keypoints.dtype,
        )
        rel = token_grid.view(1, -1, 1, 2) - keypoints.view(
            keypoints.size(0),
            1,
            self.num_keypoints,
            2,
        )
        distance = torch.linalg.norm(rel, dim=-1, keepdim=True)
        token_keypoint_bias = self.mlp(torch.cat([rel, distance], dim=-1))
        token_bias = self._aggregate(token_keypoint_bias).permute(0, 2, 1)

        if self.include_cls_token:
            cls_bias = token_bias.new_zeros(token_bias.size(0), self.num_heads, 1)
            token_bias = torch.cat([cls_bias, token_bias], dim=-1)

        if not pairwise:
            return token_bias

        return token_bias.unsqueeze(2).expand(
            -1,
            -1,
            token_bias.size(-1),
            -1,
        )
