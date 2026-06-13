from .base import BaseKANModel
from src.modules.fasterkan import FasterKAN
from src.modules.reduction import ReductionWrapper


class FasterKANModel(BaseKANModel):
    def __init__(
        self,
        layers_hidden,
        num_grids=8,
        exponent=2,
        inv_denominator=0.5,
        reduction="none",
        pool_stride=1,
        scattering_j=3,
        scattering_l=8,
        scattering_order=2,
        conv_channels=64,
        conv_layers=2,
        conv_stride_h=4,
        conv_stride_w=2,
        conv_kernel=3,
        img_height=0,
        img_width=0,
        in_chans=1,
        **kwargs,
    ):
        self.layers_hidden = layers_hidden
        self.grid_min = -1.2
        self.grid_max = 1.2
        self.num_grids = num_grids
        self.exponent = exponent
        self.inv_denominator = inv_denominator
        self.reduction = dict(
            method=reduction,
            in_chans=in_chans,
            img_height=img_height,
            img_width=img_width,
            pool_stride=pool_stride,
            scattering_j=scattering_j,
            scattering_l=scattering_l,
            scattering_order=scattering_order,
            conv_channels=conv_channels,
            conv_layers=conv_layers,
            conv_stride_h=conv_stride_h,
            conv_stride_w=conv_stride_w,
            conv_kernel=conv_kernel,
        )

    def build(self, device="cpu"):
        kan = FasterKAN(
            layers_hidden=list(self.layers_hidden),
            grid_min=self.grid_min,
            grid_max=self.grid_max,
            num_grids=self.num_grids,
            exponent=self.exponent,
            inv_denominator=self.inv_denominator,
        )
        self.model = ReductionWrapper(kan, **self.reduction).to(device)
        self.device = device
