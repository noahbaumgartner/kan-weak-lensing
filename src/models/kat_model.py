import torch
from torch.nn import functional as F

from .base import BaseKANModel
from src.modules.kat import KATVisionTransformer
from src.modules.convstem import ConvStem, StemModel


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
        native_h=None,
        native_w=None,
        stem_channels=8,
        stem_hidden_channels=None,
        stem_layers=3,
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
        # native_h/w: resolution the input is resized to *before* the conv
        # stem (defaults to img_h/img_w, i.e. no stem — the old fixed-resize
        # behaviour) so non-weak-lensing callers are unaffected.
        self.native_h = native_h if native_h is not None else img_h
        self.native_w = native_w if native_w is not None else img_w
        # unlike KKAN, patch_embed is a single ordinary Conv2d, so stem
        # out_channels is cheap here — no need to keep it small.
        self.stem_channels = stem_channels
        self.stem_hidden_channels = stem_hidden_channels
        self.stem_layers = stem_layers

    def build(self, device="cpu"):
        use_stem = (self.native_h, self.native_w) != (self.img_h, self.img_w)
        kat = KATVisionTransformer(
            img_size=(self.img_h, self.img_w),
            patch_size=self.patch_size,
            in_chans=self.stem_channels if use_stem else self.in_chans,
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
        )
        if use_stem:
            stem = ConvStem(
                in_chans=self.in_chans,
                out_channels=self.stem_channels,
                hidden_channels=self.stem_hidden_channels,
                native_h=self.native_h,
                native_w=self.native_w,
                target_h=self.img_h,
                target_w=self.img_w,
                n_layers=self.stem_layers,
            )
            self.model = StemModel(stem, kat).to(device)
        else:
            self.model = kat.to(device)
        self.device = device

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Bring any input to a (B, C, native_h, native_w) tensor.

        KAT is a KAN-ViT that patch-embeds the image; timm's PatchEmbed handles
        rectangular sizes natively. MNIST-style data is already the right size;
        the weak-lensing maps are (B, 1, 1424, 176) and are bilinearly resized
        to ``native_h x native_w`` here (a no-op unless a smaller native size
        was configured) — the learnable conv stem (see build()) then
        downsamples to ``img_h x img_w`` on its own, keeping the native 8:1
        aspect ratio; both must stay divisible by patch_size. The output_dim
        outputs double as the (Om, S8) regression head under objective=mse.
        """
        x = x.to(self.device)
        if x.dim() == 2:
            x = x.view(-1, self.in_chans, self.native_h, self.native_w)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        if x.shape[-2] != self.native_h or x.shape[-1] != self.native_w:
            x = F.interpolate(
                x, size=(self.native_h, self.native_w),
                mode="bilinear", align_corners=False,
            )
        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self._prepare_input(x))
