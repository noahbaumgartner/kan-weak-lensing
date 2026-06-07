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
        scattering_j=3,
        scattering_l=8,
        scattering_order=2,
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
            scattering_j=scattering_j,
            scattering_l=scattering_l,
            scattering_order=scattering_order,
        )

    def build(self, device="cpu"):
        kan = KAN(
            layers_hidden=list(self.layers_hidden),
            wavelet_type=self.wavelet_type,
        )
        self.model = ReductionWrapper(kan, **self.reduction).to(device)
        self.device = device
