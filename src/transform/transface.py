import math

import torch
import torch.nn as nn


class DynamicPatchAmplitudeMix(nn.Module):
    """Top-patch amplitude spectrum mixing used by TransFace-style DPAP."""

    def __init__(
        self,
        patch_grid=(12, 12),
        top_k=7,
        probability=0.2,
        alpha=1.0,
        ratio=1.0,
    ):
        super(DynamicPatchAmplitudeMix, self).__init__()
        self.patch_grid = tuple(patch_grid)
        self.top_k = top_k
        self.probability = probability
        self.alpha = alpha
        self.ratio = ratio

    def _amplitude_spectrum_mix(self, source, reference):
        lam = torch.rand((), device=source.device, dtype=source.dtype) * self.alpha
        source_fft = torch.fft.fft2(source, dim=(-2, -1))
        reference_fft = torch.fft.fft2(reference, dim=(-2, -1))

        source_amp = torch.fft.fftshift(torch.abs(source_fft), dim=(-2, -1))
        reference_amp = torch.fft.fftshift(torch.abs(reference_fft), dim=(-2, -1))
        source_phase = torch.angle(source_fft)

        height, width = source.shape[-2:]
        crop_h = max(1, int(height * math.sqrt(self.ratio)))
        crop_w = max(1, int(width * math.sqrt(self.ratio)))
        h_start = height // 2 - crop_h // 2
        w_start = width // 2 - crop_w // 2

        mixed_amp = source_amp.clone()
        mixed_amp[
            ...,
            h_start : h_start + crop_h,
            w_start : w_start + crop_w,
        ] = (
            lam
            * reference_amp[
                ...,
                h_start : h_start + crop_h,
                w_start : w_start + crop_w,
            ]
            + (1.0 - lam)
            * source_amp[
                ...,
                h_start : h_start + crop_h,
                w_start : w_start + crop_w,
            ]
        )

        mixed_amp = torch.fft.ifftshift(mixed_amp, dim=(-2, -1))
        mixed_fft = mixed_amp * torch.exp(1j * source_phase)
        return torch.fft.ifft2(mixed_fft, dim=(-2, -1)).real

    def forward(self, images, patch_scores=None):
        if images.ndim != 4:
            raise ValueError("images must have shape [B, C, H, W]")

        batch_size, _, height, width = images.shape
        grid_h, grid_w = self.patch_grid
        patch_h = height // grid_h
        patch_w = width // grid_w
        if batch_size == 0 or patch_h == 0 or patch_w == 0:
            return images

        n_patches = grid_h * grid_w
        if patch_scores is None:
            patch_scores = torch.rand(batch_size, n_patches, device=images.device)
        if patch_scores.shape != (batch_size, n_patches):
            raise ValueError(f"patch_scores must have shape {(batch_size, n_patches)}")

        output = images.clone()
        top_k = min(self.top_k, n_patches)
        top_indices = torch.topk(patch_scores, k=top_k, dim=1).indices

        for batch_index in range(batch_size):
            if torch.rand((), device=images.device).item() > self.probability:
                continue
            for patch_index in top_indices[batch_index]:
                patch_index = int(patch_index.item())
                patch_y = patch_index // grid_w
                patch_x = patch_index - patch_y * grid_w

                reference_index = int(
                    torch.randint(batch_size, (), device=images.device).item()
                )
                if batch_size > 1 and reference_index == batch_index:
                    reference_index = (reference_index + 1) % batch_size
                reference_y = int(
                    torch.randint(grid_h, (), device=images.device).item()
                )
                reference_x = int(
                    torch.randint(grid_w, (), device=images.device).item()
                )

                y0, y1 = patch_y * patch_h, (patch_y + 1) * patch_h
                x0, x1 = patch_x * patch_w, (patch_x + 1) * patch_w
                ref_y0, ref_y1 = reference_y * patch_h, (reference_y + 1) * patch_h
                ref_x0, ref_x1 = reference_x * patch_w, (reference_x + 1) * patch_w

                output[batch_index, :, y0:y1, x0:x1] = self._amplitude_spectrum_mix(
                    output[batch_index, :, y0:y1, x0:x1],
                    images[reference_index, :, ref_y0:ref_y1, ref_x0:ref_x1],
                )

        return output
