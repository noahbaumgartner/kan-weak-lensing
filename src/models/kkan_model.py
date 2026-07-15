import torch
from torch import nn
from torch.nn import functional as F

from .base import BaseKANModel
from src.modules.convkan import KAN_Convolutional_Layer
from src.modules.convkan.kanlinear import KANLinear
from src.modules.convstem import ConvStem, StemModel


class KKAN_Small(nn.Module):
    def __init__(
        self,
        grid_size: int = 5,
        img_h: int = 28,
        img_w: int = 28,
        in_chans: int = 1,
        num_classes: int = 10,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.img_h = img_h
        self.img_w = img_w

        self.conv1 = KAN_Convolutional_Layer(
            in_channels=in_chans,
            out_channels=5,
            kernel_size=(3, 3),
            grid_size=grid_size,
            padding=(0, 0),
        )

        self.conv2 = KAN_Convolutional_Layer(
            in_channels=5,
            out_channels=5,
            kernel_size=(3, 3),
            grid_size=grid_size,
            padding=(0, 0),
        )

        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.flat = nn.Flatten()

        # Track each spatial dim independently so non-square (rectangular)
        # inputs work: each stage is a 3x3 valid conv (-2) then a 2x2 pool (//2).
        def _stage_out(n: int) -> int:
            n = (n - 2) // 2
            n = (n - 2) // 2
            return n

        h_out, w_out = _stage_out(img_h), _stage_out(img_w)
        if h_out <= 0 or w_out <= 0:
            raise ValueError(
                f"input {img_h}x{img_w} too small for two (3x3 conv -> 2x2 pool) stages"
            )
        flat_dim = 5 * h_out * w_out

        self.kan1 = KANLinear(
            flat_dim,
            num_classes,
            grid_size=grid_size,
            spline_order=3,
            scale_noise=0.01,
            scale_base=1,
            scale_spline=1,
            base_activation=nn.SiLU,
            grid_eps=0.02,
            grid_range=[0, 1],
        )
        self.name = f"KKAN (Small) (gs = {grid_size})"

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.flat(x)
        x = self.kan1(x)
        return x


class KKANModel(BaseKANModel):
    def __init__(
        self,
        output_dim,
        img_h=28,
        img_w=28,
        in_chans=1,
        grid_size=5,
        native_h=None,
        native_w=None,
        stem_channels=8,
        stem_layers=2,
        **kwargs,
    ):
        self.output_dim = output_dim
        self.img_h = img_h
        self.img_w = img_w
        self.in_chans = in_chans
        self.grid_size = grid_size
        # native_h/w: resolution the input is resized to *before* the conv
        # stem (defaults to img_h/img_w, i.e. no stem — the old fixed-resize
        # behaviour) so non-weak-lensing callers are unaffected.
        self.native_h = native_h if native_h is not None else img_h
        self.native_w = native_w if native_w is not None else img_w
        self.stem_channels = stem_channels
        self.stem_layers = stem_layers

    def build(self, device="cpu"):
        use_stem = (self.native_h, self.native_w) != (self.img_h, self.img_w)
        kkan = KKAN_Small(
            grid_size=self.grid_size,
            img_h=self.img_h,
            img_w=self.img_w,
            in_chans=self.stem_channels if use_stem else self.in_chans,
            num_classes=self.output_dim,
        )
        if use_stem:
            stem = ConvStem(
                in_chans=self.in_chans,
                out_channels=self.stem_channels,
                native_h=self.native_h,
                native_w=self.native_w,
                target_h=self.img_h,
                target_w=self.img_w,
                n_layers=self.stem_layers,
            )
            self.model = StemModel(stem, kkan).to(device)
        else:
            self.model = kkan.to(device)
        self.device = device

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Bring any input to a (B, C, native_h, native_w) tensor.

        KKAN is a conv-KAN classifier. MNIST-style data is already the right
        size; the weak-lensing maps are (B, 1, 1424, 176) and are bilinearly
        resized to ``native_h x native_w`` here (a no-op unless a smaller
        native size was configured) — the learnable conv stem (see build())
        then downsamples to ``img_h x img_w`` on its own, keeping the native
        8:1 aspect ratio. The output_dim raw outputs double as the (Om, S8)
        regression head under objective=mse.
        """
        x = x.to(self.device)
        if x.dim() == 2:
            x = x.view(-1, self.in_chans, self.native_h, self.native_w)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        if x.shape[-2] != self.native_h or x.shape[-1] != self.native_w:
            x = F.interpolate(
                x, size=(self.native_h, self.native_w),
                mode="bilinear", align_corners=False,
            )
        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self._prepare_input(x))
