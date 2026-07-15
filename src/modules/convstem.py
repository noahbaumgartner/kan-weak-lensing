"""Learnable conv front end for the image-based KAN models (KKAN, KAT).

Both KKAN and KAT used to reach their working resolution with a single fixed
bilinear resize straight from the native ${dataset.img_height}x${img_width}
map down to the model's target size, blindly discarding whatever small-scale
structure fell between two pixels. A stem replaces that fixed interpolation
with a few cheap, ordinary strided Conv2d layers (BatchNorm + SiLU) that learn
what to keep while downsampling to the exact same target size — regular convs
are far cheaper than the custom KAN-conv / attention layers that follow, so
this doesn't change the resolution the expensive layers see (same RAM
footprint), only how that resolution is reached.
"""
import torch.nn as nn


def _layer_strides(native: int, target: int, n_layers: int) -> list[int]:
    """Split the native/target downscale factor evenly across n_layers.

    Requires the factor to be an exact n_layers-th power of an integer stride
    (e.g. factor=4, n_layers=2 -> stride 2 twice) so the stem lands exactly on
    (target_h, target_w) with no rounding.
    """
    if native == target:
        return [1] * n_layers
    if native % target != 0:
        raise ValueError(f"native size {native} is not a multiple of target size {target}")
    factor = native // target
    stride = round(factor ** (1.0 / n_layers))
    if stride < 1 or stride**n_layers != factor:
        raise ValueError(
            f"downscale factor {factor} (={native}/{target}) is not an exact "
            f"{n_layers}-layer power of an integer stride; adjust stem_layers"
        )
    return [stride] * n_layers


class ConvStem(nn.Module):
    """Strided Conv2d -> BatchNorm -> SiLU stack, ending at (target_h, target_w).

    Stride is computed independently per axis (kernel=3, padding=1 exactly
    halves/divides an evenly-divisible input at each layer), so a rectangular
    downscale factor — like the elongated weak-lensing maps — still lands
    exactly on (target_h, target_w).
    """

    def __init__(
        self,
        in_chans: int,
        out_channels: int,
        native_h: int,
        native_w: int,
        target_h: int,
        target_w: int,
        n_layers: int = 2,
        kernel: int = 3,
    ):
        super().__init__()
        strides_h = _layer_strides(native_h, target_h, n_layers)
        strides_w = _layer_strides(native_w, target_w, n_layers)

        layers: list[nn.Module] = []
        c_in = in_chans
        pad = kernel // 2
        for sh, sw in zip(strides_h, strides_w):
            layers += [
                nn.Conv2d(c_in, out_channels, kernel_size=kernel, stride=(sh, sw), padding=pad),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(),
            ]
            c_in = out_channels
        self.net = nn.Sequential(*layers)
        self.out_channels = out_channels

    def forward(self, x):
        return self.net(x)


class StemModel(nn.Module):
    """Prepend a :class:`ConvStem` to an inner model as a single nn.Module, so
    the stem's parameters are picked up automatically wherever the wrapping
    ``BaseKANModel`` iterates ``self.model.parameters()`` (optimizer
    construction, ``parameter_count()``, checkpointing, ...)."""

    def __init__(self, stem: ConvStem, inner: nn.Module):
        super().__init__()
        self.stem = stem
        self.inner = inner

    def forward(self, x):
        return self.inner(self.stem(x))
