"""
Swin Transformer + Intra-Stage Block AttnRes (v3)
Strictly follows paper Figure 2 logic (arXiv: 2603.15031).

Per-layer forward (paper Fig.2 lines 16-38):
    1. partial_block = hidden_states
    2. h = AttnRes(blocks, partial_block)       # before attn
    3. if block_boundary: blocks.append(partial_block); partial_block = None
    4. attn_out = attn(norm(h))
    5. partial_block += attn_out
    6. h = AttnRes(blocks, partial_block)       # before mlp
    7. mlp_out = mlp(norm(h))
    8. partial_block += mlp_out

timm Swin verified structure:
    stage.forward: x = downsample(x); x = blocks(x)
    SwinBlock.forward:
        x = x + drop_path1(_attn(norm1(x)))
        x = x + drop_path2(mlp(norm2(x)))
    head() does global pool internally.
"""
import torch
import torch.nn as nn
import timm
from .attnres import BlockAttnRes


SWIN_CONFIGS = {
    'swin_tiny': {
        'name': 'swin_tiny_patch4_window7_224',
        'dims': [96, 192, 384, 768],
        'blocks': [2, 2, 6, 2],        # 12 SwinBlocks = 24 sub-layers
    },
    'swin_small': {
        'name': 'swin_small_patch4_window7_224',
        'dims': [96, 192, 384, 768],
        'blocks': [2, 2, 18, 2],       # 24 SwinBlocks = 48 sub-layers
    },
    'swin_base': {
        'name': 'swin_base_patch4_window7_224',
        'dims': [128, 256, 512, 1024],
        'blocks': [2, 2, 18, 2],       # 24 SwinBlocks = 48 sub-layers
    },
}


class SwinWithAttnRes(nn.Module):
    """
    Swin Transformer with Intra-Stage Block AttnRes.

    Follows paper Fig.2 exactly:
    - Each sub-layer (attn, mlp) has its own AttnRes applied BEFORE it
    - partial_block accumulates pure sub-layer outputs
    - Block boundary checked BEFORE each sub-layer (not after)
    - Every block_size sub-layers, partial_block → blocks list, then reset
    """

    def __init__(self, arch='swin_tiny', num_classes=100, pretrained=False,
                 drop_path_rate=0.0, block_size=6):
        """
        Args:
            arch: 'swin_tiny', 'swin_small', or 'swin_base'
            block_size: sub-layers per block (6 = every 3 SwinBlocks)
        """
        super().__init__()
        assert arch in SWIN_CONFIGS, f"Unknown arch: {arch}"
        cfg = SWIN_CONFIGS[arch]

        self.swin = timm.create_model(
            cfg['name'], pretrained=pretrained, num_classes=num_classes,
            drop_path_rate=drop_path_rate)
        self.stage_dims = cfg['dims']
        self.stage_blocks = cfg['blocks']
        self.block_size = block_size

        # One AttnRes per sub-layer + one final per stage
        self.attn_res = nn.ModuleDict()
        for si, (num_swin_blocks, dim) in enumerate(zip(self.stage_blocks, self.stage_dims)):
            num_sublayers = num_swin_blocks * 2  # attn + mlp
            for li in range(num_sublayers):
                self.attn_res[f"s{si}_l{li}"] = BlockAttnRes(dim)
            self.attn_res[f"s{si}_final"] = BlockAttnRes(dim)

    def forward(self, x):
        x = self.swin.patch_embed(x)
        if hasattr(self.swin, 'pos_drop'):
            x = self.swin.pos_drop(x)

        for si, stage in enumerate(self.swin.layers):
            # Downsample first (Identity for stage 0, PatchMerging for 1-3)
            x = stage.downsample(x)

            # b0 = stage input after downsample (paper Eq.6: b0 = h1)
            blocks = [x.clone()]
            partial_block = None
            sublayer_count = 0

            for bi, swin_block in enumerate(stage.blocks):

                # ══ Sub-layer 1: Attention ══

                # Paper Line 19: AttnRes before attn
                pb = partial_block if partial_block is not None else torch.zeros_like(x)
                h = self.attn_res[f"s{si}_l{sublayer_count}"](blocks, pb)

                # Paper Line 23-25: block boundary check BEFORE attn
                sublayer_count += 1
                if sublayer_count % self.block_size == 0 and partial_block is not None:
                    blocks.append(partial_block)
                    partial_block = None

                # Paper Line 28: attn
                attn_out = swin_block.drop_path1(swin_block._attn(swin_block.norm1(h)))

                # Paper Line 29: accumulate
                partial_block = attn_out if partial_block is None else partial_block + attn_out

                # ══ Sub-layer 2: MLP ══

                # Paper Line 32: AttnRes before mlp
                # partial_block already includes attn_out, so h2 naturally has attn info
                h2 = self.attn_res[f"s{si}_l{sublayer_count}"](blocks, partial_block)

                # Block boundary check before mlp
                sublayer_count += 1
                if sublayer_count % self.block_size == 0 and partial_block is not None:
                    blocks.append(partial_block)
                    partial_block = None

                # Paper Line 35: mlp (directly on h2, no extra attn_out addition)
                mlp_out = swin_block.drop_path2(swin_block.mlp(swin_block.norm2(h2)))

                # Paper Line 36: accumulate
                partial_block = mlp_out if partial_block is None else partial_block + mlp_out

            # Final AttnRes for stage output
            pb = partial_block if partial_block is not None else torch.zeros_like(x)
            x = self.attn_res[f"s{si}_final"](blocks, pb)

        # Head (norm + global_pool + linear)
        x = self.swin.norm(x)
        x = self.swin.head(x)
        return x
