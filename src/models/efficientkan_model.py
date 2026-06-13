from .base import BaseKANModel
from src.modules.efficientkan import KAN
from src.modules.reduction import ReductionWrapper


class EfficientKANModel(BaseKANModel):
    """B-spline KAN (Blealtan/efficient-kan) with the shared image reduction.

    Closest of the MLP-style models to the original KAN paper: a real B-spline
    basis (``grid_size`` knots, ``spline_order`` k) plus an L1/entropy
    regularisation on the spline weights (see ``regularization_loss``).
    """

    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        k=3,
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
        self.grid_size = grid_size
        self.k = k
        # Image -> vector reduction applied to 2D inputs (e.g. weak_lensing)
        # before flattening. Defaults to a no-op passthrough for tabular/1D.
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
