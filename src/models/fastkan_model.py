import torch.nn as nn

from .base import BaseKANModel
from src.modules.fastkan import FastKAN, FastKANLayer
from src.modules.reduction import ReductionWrapper


class FastKANModel(BaseKANModel):
    def __init__(
        self,
        layers_hidden,
        num_grids=8,
        reduction="none",
        pool_stride=1,
        conv_channels=64,
        conv_layers=2,
        conv_stride_h=4,
        conv_stride_w=2,
        conv_kernel=3,
        scattering_J=3,
        scattering_L=8,
        scattering_order=2,
        img_height=0,
        img_width=0,
        in_chans=1,
        **kwargs,
    ):
        self.layers_hidden = layers_hidden
        self.grid_min = -2.0
        self.grid_max = 2.0
        self.num_grids = num_grids
        self.reduction = dict(
            method=reduction,
            in_chans=in_chans,
            img_height=img_height,
            img_width=img_width,
            pool_stride=pool_stride,
            conv_channels=conv_channels,
            conv_layers=conv_layers,
            conv_stride_h=conv_stride_h,
            conv_stride_w=conv_stride_w,
            conv_kernel=conv_kernel,
            scattering_J=scattering_J,
            scattering_L=scattering_L,
            scattering_order=scattering_order,
        )

    def build(self, device="cpu"):
        layers_hidden = list(self.layers_hidden)
        needs_custom = any(d == 1 for d in layers_hidden[:-1])
        if needs_custom:
            kan = FastKAN(layers_hidden=[2, 1], num_grids=self.num_grids)
            kan.layers = nn.ModuleList(
                [
                    FastKANLayer(
                        in_dim,
                        out_dim,
                        grid_min=self.grid_min,
                        grid_max=self.grid_max,
                        num_grids=self.num_grids,
                        use_layernorm=in_dim > 1,
                    )
                    for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
                ]
            )
        else:
            kan = FastKAN(
                layers_hidden=layers_hidden,
                grid_min=self.grid_min,
                grid_max=self.grid_max,
                num_grids=self.num_grids,
            )
        self.model = ReductionWrapper(kan, **self.reduction).to(device)
        self.device = device
