"""
Swin Transformer + Intra-Stage Block AttnRes
Supports: Swin-T, Swin-S, Swin-B

timm Swin internal structure (verified):
    Each stage.forward: x = downsample(x); x = blocks(x)
    Downsample is Identity for stage 0, PatchMerging for stages 1-3.
    head() does global pool internally — do NOT manually mean().
"""
import torch
import torch.nn as nn
import timm
from .attnres import BlockAttnRes


SWIN_CONFIGS = {
    'swin_tiny': {
        'name': 'swin_tiny_patch4_window7_224',
        'dims': [96, 192, 384, 768],
        'blocks': [2, 2, 6, 2],       # 12 layers total
    },
    'swin_small': {
        'name': 'swin_small_patch4_window7_224',
        'dims': [96, 192, 384, 768],
        'blocks': [2, 2, 18, 2],      # 24 layers total
    },
    'swin_base': {
        'name': 'swin_base_patch4_window7_224',
        'dims': [128, 256, 512, 1024],
        'blocks': [2, 2, 18, 2],      # 24 layers total
    },
}


class SwinWithAttnRes(nn.Module):
    """Swin Transformer with Intra-Stage Block AttnRes."""

    def __init__(self, arch='swin_tiny', num_classes=100, pretrained=False,
                 drop_path_rate=0.0):
        super().__init__()
        assert arch in SWIN_CONFIGS, f"Unknown arch: {arch}. Choose from {list(SWIN_CONFIGS.keys())}"
        cfg = SWIN_CONFIGS[arch]

        self.swin = timm.create_model(
            cfg['name'],
            pretrained=pretrained,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
        )
        self.stage_dims = cfg['dims']
        self.stage_blocks = cfg['blocks']

        # Create AttnRes modules: one per block + one final per stage
        self.attn_res = nn.ModuleDict()
        for si, (num_blocks, dim) in enumerate(zip(self.stage_blocks, self.stage_dims)):
            for bi in range(num_blocks):
                self.attn_res[f"s{si}_b{bi}"] = BlockAttnRes(dim)
            self.attn_res[f"s{si}_final"] = BlockAttnRes(dim)

    def forward(self, x):
        # Patch embedding
        x = self.swin.patch_embed(x)
        if hasattr(self.swin, 'pos_drop'):
            x = self.swin.pos_drop(x)

        # Process each stage
        for si, stage in enumerate(self.swin.layers):
            # Downsample first (Identity for stage 0, PatchMerging for 1-3)
            x = stage.downsample(x)

            # Intra-stage AttnRes over blocks
            blocks = [x.clone()]       # b0 = stage input after downsample
            partial_block = None

            for bi, swin_block in enumerate(stage.blocks):
                pb = partial_block if partial_block is not None else torch.zeros_like(x)
                h = self.attn_res[f"s{si}_b{bi}"](blocks, pb)

                out = swin_block(h)    # SwinBlock has internal residual
                delta = out - h        # extract layer's contribution

                partial_block = delta if partial_block is None else partial_block + delta

            # Final AttnRes to produce stage output
            pb = partial_block if partial_block is not None else torch.zeros_like(x)
            x = self.attn_res[f"s{si}_final"](blocks, pb)

        # Head (norm + global_pool + linear, all inside self.swin)
        x = self.swin.norm(x)
        x = self.swin.head(x)
        return x
