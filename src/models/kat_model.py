import torch
from torch.nn import functional as F

from .base import BaseKANModel
from src.modules.kat import KATVisionTransformer


class KATModel(BaseKANModel):
    def __init__(
        self,
        output_dim,
        img_h=28,
        img_w=28,
        patch_size=4,
        in_chans=1,
        embed_dim=64,
        depth=4,
        num_heads=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        proj_drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        global_pool="token",
        act_init="gelu",
        weight_init="kan",
        **kwargs,
    ):
        self.output_dim = output_dim
        self.img_h = img_h
        self.img_w = img_w
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.proj_drop_rate = proj_drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.global_pool = global_pool
        self.act_init = act_init
        self.weight_init = weight_init

    def build(self, device="cpu"):
        self.model = KATVisionTransformer(
            img_size=(self.img_h, self.img_w),
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            num_classes=self.output_dim,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            drop_rate=self.drop_rate,
            proj_drop_rate=self.proj_drop_rate,
            attn_drop_rate=self.attn_drop_rate,
            drop_path_rate=self.drop_path_rate,
            global_pool=self.global_pool,
            act_init=self.act_init,
            weight_init=self.weight_init,
        ).to(device)
        self.device = device

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Bring any input to a (B, C, img_h, img_w) tensor.

        KAT is a KAN-ViT that patch-embeds the image; timm's PatchEmbed handles
        rectangular sizes natively. MNIST-style data is already the right size;
        the weak-lensing maps are (B, 1, 1424, 176) and are bilinearly resized
        to ``img_h x img_w`` here. img_h/img_w keep the native 8:1 aspect ratio
        (default 178x22 = 1424x176 / 8), so the map is downscaled without
        distortion; both must be divisible by patch_size. The output_dim
        outputs double as the (Om, S8) regression head under objective=mse.
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
