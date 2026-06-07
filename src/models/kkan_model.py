import torch
from torch import nn

from .base import BaseKANModel
from src.modules.convkan import KAN_Convolutional_Layer
from src.modules.convkan.kanlinear import KANLinear


class KKAN_Small(nn.Module):
    def __init__(
        self,
        grid_size: int = 5,
        img_size: int = 28,
        in_chans: int = 1,
        num_classes: int = 10,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.img_size = img_size

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

        s = (img_size - 2) // 2
        s = (s - 2) // 2
        if s <= 0:
            raise ValueError(
                f"img_size={img_size} too small for two (3x3 conv -> 2x2 pool) stages"
            )
        flat_dim = 5 * s * s

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
    def __init__(self, num_classes, img_size=28, in_chans=1, grid_size=5, **kwargs):
        self.num_classes = num_classes
        self.img_size = img_size
        self.in_chans = in_chans
        self.grid_size = grid_size

    def build(self, device="cpu"):
        self.model = KKAN_Small(
            grid_size=self.grid_size,
            img_size=self.img_size,
            in_chans=self.in_chans,
            num_classes=self.num_classes,
        ).to(device)
        self.device = device

    def predict(self, x: torch.Tensor, update_grid: bool = False) -> torch.Tensor:
        x = x.to(self.device)
        if x.dim() == 2:
            x = x.view(-1, self.in_chans, self.img_size, self.img_size)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        return self.model(x)
