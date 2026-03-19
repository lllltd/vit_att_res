"""
Swin Transformer + Intra-Stage Block AttnRes
Based on timm's Swin-T implementation.

timm Swin-T internal structure (verified):
    patch_embed: [B,3,224,224] -> [B,56,56,96]
    Stage 0: downsample=Identity     -> 2 blocks -> [B,56,56,96]
    Stage 1: downsample=PatchMerging -> 2 blocks -> [B,28,28,192]
    Stage 2: downsample=PatchMerging -> 6 blocks -> [B,14,14,384]
    Stage 3: downsample=PatchMerging -> 2 blocks -> [B,7,7,768]
    norm -> head (head does global pool internally)
"""
import torch
import torch.nn as nn
import timm
from .attnres import BlockAttnRes


class SwinWithAttnRes(nn.Module):
    """Swin-T with Intra-Stage Block AttnRes."""

    # Swin-T architecture constants
    STAGE_DIMS = [96, 192, 384, 768]
    STAGE_BLOCKS = [2, 2, 6, 2]

    def __init__(self, num_classes=100, pretrained=False):
        super().__init__()
        self.swin = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=pretrained,
            num_classes=num_classes,
        )

        # Create AttnRes modules: one per block + one final per stage
        self.attn_res = nn.ModuleDict()
        for si, (num_blocks, dim) in enumerate(zip(self.STAGE_BLOCKS, self.STAGE_DIMS)):
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
            blocks = [x.clone()]      # b0 = stage input after downsample
            partial_block = None

            for bi, swin_block in enumerate(stage.blocks):
                pb = partial_block if partial_block is not None else torch.zeros_like(x)
                h = self.attn_res[f"s{si}_b{bi}"](blocks, pb)

                out = swin_block(h)   # SwinBlock has internal residual
                delta = out - h       # extract layer's contribution

                partial_block = delta if partial_block is None else partial_block + delta

            # Final AttnRes to produce stage output
            pb = partial_block if partial_block is not None else torch.zeros_like(x)
            x = self.attn_res[f"s{si}_final"](blocks, pb)

        # Head (norm + global_pool + linear, all inside self.swin)
        x = self.swin.norm(x)
        x = self.swin.head(x)
        return x
