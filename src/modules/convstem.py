"""Learnable conv front end for the image-based KAN models (KKAN, KAT).

Replaces a fixed bilinear resize from the native map size down to the model's
target resolution with a few cheap, ordinary strided Conv2d layers that learn
what to keep while downsampling.
"""
import torch.nn as nn


def _layer_strides(native: int, target: int, n_layers: int) -> list[int]:
    """Split the native/target downscale factor evenly across n_layers (must divide exactly)."""
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

    Keep out_channels small for KKAN: KAN_Convolutional_Layer's cost scales
    multiplicatively with it (one KANLinear per in x out channel pair).
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
        hidden_channels: int | None = None,
    ):
        super().__init__()
        strides_h = _layer_strides(native_h, target_h, n_layers)
        strides_w = _layer_strides(native_w, target_w, n_layers)
        hidden = hidden_channels if hidden_channels is not None else out_channels

        layers: list[nn.Module] = []
        c_in = in_chans
        pad = kernel // 2
        for i, (sh, sw) in enumerate(zip(strides_h, strides_w)):
            c_out = out_channels if i == n_layers - 1 else hidden
            layers += [
                nn.Conv2d(c_in, c_out, kernel_size=kernel, stride=(sh, sw), padding=pad),
                nn.BatchNorm2d(c_out),
                nn.SiLU(),
            ]
            c_in = c_out
        self.net = nn.Sequential(*layers)
        self.out_channels = out_channels

    def forward(self, x):
        return self.net(x)


class StemModel(nn.Module):
    """Prepend a ConvStem to an inner model as a single nn.Module, so its params are picked up automatically."""

    def __init__(self, stem: ConvStem, inner: nn.Module):
        super().__init__()
        self.stem = stem
        self.inner = inner

    def forward(self, x):
        return self.inner(self.stem(x))
