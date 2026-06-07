import torch

from .base import BaseKANModel
from src.modules.kat import KATVisionTransformer


class KATModel(BaseKANModel):
    def __init__(
        self,
        num_classes,
        img_size=28,
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
        self.num_classes = num_classes
        self.img_size = img_size
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
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            num_classes=self.num_classes,
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

    def predict(self, x: torch.Tensor, update_grid: bool = False) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(-1, self.in_chans, self.img_size, self.img_size)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        return self.model(x)
