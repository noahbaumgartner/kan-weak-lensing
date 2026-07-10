from .base import BaseKANModel
from src.modules.efficientkan import KAN
from src.modules.reduction import ReductionWrapper


class EfficientKANModel(BaseKANModel):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        k=3,
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
        self.grid_size = grid_size
        self.k = k
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
        kan = KAN(
            layers_hidden=list(self.layers_hidden),
            grid_size=self.grid_size,
            spline_order=self.k,
            grid_range=[-3, 3],
        )
        self.model = ReductionWrapper(kan, **self.reduction).to(device)
        self.device = device

    def regularization_loss(self):
        return self.model.regularization_loss()
