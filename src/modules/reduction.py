"""Image -> vector reductions shared by the MLP-style KAN models.

A KAN's first-layer width is fixed at config-compose time from
``dataset.input_dim`` while the actual reduced vector is produced at runtime by
``ReductionWrapper``. Both go through :func:`reduced_dim` so they always agree:
change the reduction method/params and the config width and runtime module move
together.

Available methods (set via ``dataset.reduction``):

* ``avgpool``        — fixed-stride average pooling, then flatten.
* ``conv``           — a small learnable strided-conv stack with an asymmetric
  stride (larger along the bigger image dimension) followed by global average
  pooling, leaving one feature per output channel. (Dozenten-Vorschlag 1.)
* ``scattering``      — fixed (non-learnable) wavelet scattering transform
  (kymatio, https://www.kymat.io/), then global average pooling over the
  spatial scattering-coefficient map, leaving one feature per
  scale/angle/order channel. Translation-invariant, deformation-stable
  texture features — no training needed for the reduction itself.
* ``none``           — flatten the full image (or pass tabular input through).
"""
import torch
import torch.nn as nn

# kymatio.torch eagerly imports the 3D (HarmonicScattering3D) frontend, which
# pulls in scipy.special.sph_harm — removed in scipy>=1.15 (renamed
# sph_harm_y) and unrelated to the 2D transform we use. Importing the 2D
# frontend directly sidesteps that broken import chain.
from kymatio.scattering2d.frontend.torch_frontend import ScatteringTorch2D as Scattering2D


def _scattering_n_coeffs(J: int, L: int, order: int) -> int:
    """Number of scattering channels kymatio produces for J scales, L angles,
    up to ``order`` (1 or 2). Matches kymatio's own coefficient count: 1 (order
    0, the lowpass) + L*J (order 1) + L^2 * J*(J-1)/2 (order 2, one path per
    scale pair j1<j2 x angle pair)."""
    n = 1 + L * J
    if order >= 2:
        n += L * L * J * (J - 1) // 2
    return n


def reduced_dim(
    method: str,
    h: int,
    w: int,
    in_chans: int = 1,
    pool_stride: int = 1,
    conv_channels: int = 64,
    scattering_J: int = 3,
    scattering_L: int = 8,
    scattering_order: int = 2,
) -> int:
    """Flat feature count produced by ``method`` on an ``(in_chans, h, w)`` image."""
    h, w, in_chans = int(h), int(w), int(in_chans)
    if method == "avgpool":
        return in_chans * (h // int(pool_stride)) * (w // int(pool_stride))
    if method == "conv":
        # The conv stack ends in global average pooling, so the feature count is
        # just the number of output channels (independent of input H/W/stride).
        return int(conv_channels)
    if method == "scattering":
        # Global-avg-pooled over space (like conv), so independent of H/W too —
        # just in_chans scattering channels per input channel.
        n_coeffs = _scattering_n_coeffs(
            int(scattering_J), int(scattering_L), int(scattering_order)
        )
        return in_chans * n_coeffs
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
        conv_channels: int = 64,
        conv_layers: int = 2,
        conv_stride_h: int = 4,
        conv_stride_w: int = 2,
        conv_kernel: int = 3,
        scattering_J: int = 3,
        scattering_L: int = 8,
        scattering_order: int = 2,
    ):
        super().__init__()
        self.method = method
        self.kan = kan
        self.flatten = nn.Flatten()
        self.pool = None
        self.conv = None
        self.scattering = None

        if method == "avgpool":
            self.pool = (
                nn.AvgPool2d(kernel_size=pool_stride, stride=pool_stride)
                if pool_stride > 1
                else nn.Identity()
            )
        elif method == "conv":
            # Learnable strided-conv encoder. The stride is asymmetric — larger
            # along whichever spatial axis is bigger — to compress the elongated
            # 1424x176 maps roughly isotropically before global average pooling.
            self.conv = _build_conv_encoder(
                in_chans=int(in_chans),
                out_channels=int(conv_channels),
                n_layers=int(conv_layers),
                stride_h=int(conv_stride_h),
                stride_w=int(conv_stride_w),
                kernel=int(conv_kernel),
            )
        elif method == "scattering":
            # Fixed (non-learnable) filters, sized to the exact input shape —
            # shape is baked in at construction, like the conv encoder's stride.
            self.scattering = Scattering2D(
                J=int(scattering_J),
                shape=(int(img_height), int(img_width)),
                L=int(scattering_L),
                max_order=int(scattering_order),
            )
        else:  # none -> passthrough
            self.pool = nn.Identity()

    def forward(self, x):
        if x.dim() == 4:  # (B, C, H, W)
            if self.method == "conv":
                # (B, C, H, W) -> conv stack -> global-avg -> (B, conv_channels, 1, 1)
                x = self.conv(x)
            elif self.method == "scattering":
                # (B, C, H, W) -> (B, C, n_coeffs, H', W') -> global-avg over
                # space -> (B, C, n_coeffs), same "one feature per channel"
                # shape as conv's global-avg-pooled output.
                x = self.scattering(x).mean(dim=(-1, -2))
            else:
                x = self.pool(x)
        return self.kan(self.flatten(x))

    def regularization_loss(self, *args, **kwargs):
        return self.kan.regularization_loss(*args, **kwargs)


def _build_conv_encoder(
    in_chans: int,
    out_channels: int,
    n_layers: int,
    stride_h: int,
    stride_w: int,
    kernel: int,
) -> nn.Module:
    """Strided Conv2d -> BatchNorm -> SiLU stack ending in global average pooling."""
    layers: list[nn.Module] = []
    c_in = in_chans
    pad = kernel // 2
    for _ in range(max(1, n_layers)):
        layers += [
            nn.Conv2d(
                c_in,
                out_channels,
                kernel_size=kernel,
                stride=(stride_h, stride_w),
                padding=pad,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        ]
        c_in = out_channels
    layers.append(nn.AdaptiveAvgPool2d(1))  # asymmetric input -> (C, 1, 1)
    return nn.Sequential(*layers)
