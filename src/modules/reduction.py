"""Image -> vector reductions shared by the MLP-style KAN models.

A KAN's first-layer width is fixed at config-compose time from
``dataset.input_dim`` while the actual reduced vector is produced at runtime by
``ReductionWrapper``. Both go through :func:`reduced_dim` so they always agree:
change the reduction method/params and the config width and runtime module move
together.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def num_scattering_coeffs(j: int, l: int, order: int) -> int:
    """Number of 2D scattering channels for ``J`` scales, ``L`` angles, up to ``order``."""
    j, l, order = int(j), int(l), int(order)
    count = 1  # order 0 (low-pass)
    if order >= 1:
        count += l * j
    if order >= 2:
        count += l * l * j * (j - 1) // 2
    return count


def reduced_dim(
    method: str,
    h: int,
    w: int,
    in_chans: int = 1,
    pool_stride: int = 1,
    scattering_j: int = 3,
    scattering_l: int = 8,
    scattering_order: int = 2,
) -> int:
    """Flat feature count produced by ``method`` on an ``(in_chans, h, w)`` image."""
    h, w, in_chans = int(h), int(w), int(in_chans)
    if method == "avgpool":
        return in_chans * (h // int(pool_stride)) * (w // int(pool_stride))
    if method == "kymatio":
        # Spatial output is global-average-pooled away, leaving one feature per
        # scattering coefficient per input channel.
        return in_chans * num_scattering_coeffs(scattering_j, scattering_l, scattering_order)
    if method == "none":
        return in_chans * h * w
    raise ValueError(f"unknown reduction method: {method!r}")


class ReductionWrapper(nn.Module):
    """Reduce 2D image inputs to a flat vector, then run the inner KAN.

    For already-flat (tabular / functional) inputs the reduction is skipped and
    the input is passed straight through, so non-image runs are unaffected.
    """

    def __init__(
        self,
        kan: nn.Module,
        method: str = "none",
        in_chans: int = 1,
        img_height: int = 0,
        img_width: int = 0,
        pool_stride: int = 1,
        scattering_j: int = 3,
        scattering_l: int = 8,
        scattering_order: int = 2,
    ):
        super().__init__()
        self.method = method
        self.kan = kan
        self.flatten = nn.Flatten()

        if method == "avgpool":
            self.pool = (
                nn.AvgPool2d(kernel_size=pool_stride, stride=pool_stride)
                if pool_stride > 1
                else nn.Identity()
            )
            self.scattering = None
        elif method == "kymatio":
            # Import the 2D torch frontend directly: ``kymatio.torch`` also pulls
            # in the 3D filter bank, which needs scipy.special.sph_harm (removed
            # in scipy >= 1.15) and would break the import.
            from kymatio.scattering2d.frontend.torch_frontend import (
                ScatteringTorch2D as Scattering2D,
            )

            self.pool = None
            self.scattering = Scattering2D(
                J=int(scattering_j),
                shape=(int(img_height), int(img_width)),
                L=int(scattering_l),
                max_order=int(scattering_order),
            )
        else:  # none -> passthrough
            self.pool = nn.Identity()
            self.scattering = None

    def forward(self, x):
        if x.dim() == 4:  # (B, C, H, W)
            if self.method == "kymatio":
                # (B, C, H, W) -> (B, C, P, h', w') -> global-avg -> (B, C*P)
                x = self.scattering(x)
                x = x.mean(dim=(-1, -2))
            else:
                x = self.pool(x)
        return self.kan(self.flatten(x))

    def regularization_loss(self, *args, **kwargs):
        return self.kan.regularization_loss(*args, **kwargs)
