from .base import BaseKANModel
from src.modules.wavkan.kan import KAN
from src.modules.reduction import ReductionWrapper


class WavKANModel(BaseKANModel):
    def __init__(
        self,
        layers_hidden,
        wavelet_type="mexican_hat",
        reduction="none",
        pool_stride=1,
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
        self.wavelet_type = wavelet_type
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
        )

    def build(self, device="cpu"):
        kan = KAN(
            layers_hidden=list(self.layers_hidden),
            wavelet_type=self.wavelet_type,
        )
        self.model = ReductionWrapper(kan, **self.reduction).to(device)
        self.device = device
