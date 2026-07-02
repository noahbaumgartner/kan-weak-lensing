import torch
from torch import nn
from torch.nn import functional as F

from .base import BaseKANModel
from src.modules.convkan import KAN_Convolutional_Layer
from src.modules.convkan.kanlinear import KANLinear


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
    def __init__(self, output_dim, img_h=28, img_w=28, in_chans=1, grid_size=5, **kwargs):
        self.output_dim = output_dim
        self.img_h = img_h
        self.img_w = img_w
        self.in_chans = in_chans
        self.grid_size = grid_size

    def build(self, device="cpu"):
        self.model = KKAN_Small(
            grid_size=self.grid_size,
            img_h=self.img_h,
            img_w=self.img_w,
            in_chans=self.in_chans,
            num_classes=self.output_dim,
        ).to(device)
        self.device = device

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Bring any input to a (B, C, img_h, img_w) tensor.

        KKAN is a conv-KAN classifier. MNIST-style data is already the right
        size; the weak-lensing maps are (B, 1, 1424, 176) and are bilinearly
        resized to ``img_h x img_w`` here. img_h/img_w keep the native 8:1
        aspect ratio (default 178x22 = 1424x176 / 8), so the map is downscaled
        without distortion. The output_dim raw outputs double as the (Om, S8)
        regression head under objective=mse.
        """
        x = x.to(self.device)
        if x.dim() == 2:
            x = x.view(-1, self.in_chans, self.img_h, self.img_w)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        if x.shape[-2] != self.img_h or x.shape[-1] != self.img_w:
            x = F.interpolate(
                x, size=(self.img_h, self.img_w),
                mode="bilinear", align_corners=False,
            )
        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self._prepare_input(x))
