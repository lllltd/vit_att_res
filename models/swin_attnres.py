"""
Swin Transformer + Intra-Stage Block AttnRes (v4)
Strictly follows paper Figure 2 logic (arXiv: 2603.15031).

Supported architectures:
    swin_tiny:  (2,2, 6,2) = 12 layers, 28M params
    swin_small: (2,2,18,2) = 24 layers, 49M params
    swin_deep:  (2,2,30,2) = 36 layers, 70M params  (custom, same width as Swin-T)
    swin_base:  (2,2,18,2) = 24 layers, 88M params  (wider channels)

Per-layer forward (paper Fig.2 lines 16-38):
    1. partial_block = hidden_states
    2. h = AttnRes(blocks, partial_block)       # before attn
    3. if block_boundary: blocks.append(partial_block); partial_block = None
    4. attn_out = attn(norm(h))
    5. partial_block += attn_out
    6. h = AttnRes(blocks, partial_block)       # before mlp
    7. mlp_out = mlp(norm(h))
    8. partial_block += mlp_out
"""
import torch
import torch.nn as nn
import timm
from .attnres import BlockAttnRes


SWIN_CONFIGS = {
    'swin_tiny': {
        'name': 'swin_tiny_patch4_window7_224',
        'dims': [96, 192, 384, 768],
        'blocks': [2, 2, 6, 2],        # 12 layers
        'num_heads': [3, 6, 12, 24],
    },
    'swin_small': {
        'name': 'swin_small_patch4_window7_224',
        'dims': [96, 192, 384, 768],
        'blocks': [2, 2, 18, 2],       # 24 layers
        'num_heads': [3, 6, 12, 24],
    },
    'swin_deep': {
        'name': '__custom__',           # not a timm preset, built manually
        'dims': [96, 192, 384, 768],
        'blocks': [2, 2, 30, 2],       # 36 layers
        'num_heads': [3, 6, 12, 24],
    },
    'swin_base': {
        'name': 'swin_base_patch4_window7_224',
        'dims': [128, 256, 512, 1024],
        'blocks': [2, 2, 18, 2],       # 24 layers
        'num_heads': [4, 8, 16, 32],
    },
}


def _build_swin_backbone(arch, num_classes=100, pretrained=False, drop_path_rate=0.0):
    """Build a Swin backbone, using timm presets or custom config."""
    cfg = SWIN_CONFIGS[arch]
    if cfg['name'] != '__custom__':
        return timm.create_model(
            cfg['name'], pretrained=pretrained, num_classes=num_classes,
            drop_path_rate=drop_path_rate)
    else:
        # Custom architecture (e.g. swin_deep)
        from timm.models.swin_transformer import SwinTransformer
        return SwinTransformer(
            img_size=224, patch_size=4, in_chans=3, num_classes=num_classes,
            embed_dim=cfg['dims'][0],
            depths=cfg['blocks'],
            num_heads=cfg['num_heads'],
            window_size=7, mlp_ratio=4.0, qkv_bias=True,
            drop_path_rate=drop_path_rate)


class SwinWithAttnRes(nn.Module):
    """
    Swin Transformer with Intra-Stage Block AttnRes.

    Follows paper Fig.2 exactly:
    - Each sub-layer (attn, mlp) has its own AttnRes applied BEFORE it
    - partial_block accumulates pure sub-layer outputs
    - Block boundary checked BEFORE each sub-layer
    - Every block_size sub-layers, partial_block -> blocks list, then reset
    """

    def __init__(self, arch='swin_tiny', num_classes=100, pretrained=False,
                 drop_path_rate=0.0, block_size=6):
        super().__init__()
        assert arch in SWIN_CONFIGS, f"Unknown arch: {arch}"
        cfg = SWIN_CONFIGS[arch]

        self.swin = _build_swin_backbone(arch, num_classes, pretrained, drop_path_rate)
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
            x = stage.downsample(x)

            blocks = [x.clone()]
            partial_block = None
            sublayer_count = 0

            for bi, swin_block in enumerate(stage.blocks):

                # ══ Sub-layer 1: Attention ══
                pb = partial_block if partial_block is not None else torch.zeros_like(x)
                h = self.attn_res[f"s{si}_l{sublayer_count}"](blocks, pb)

                sublayer_count += 1
                if sublayer_count % self.block_size == 0 and partial_block is not None:
                    blocks.append(partial_block)
                    partial_block = None

                attn_out = swin_block.drop_path1(swin_block._attn(swin_block.norm1(h)))
                partial_block = attn_out if partial_block is None else partial_block + attn_out

                # ══ Sub-layer 2: MLP ══
                h2 = self.attn_res[f"s{si}_l{sublayer_count}"](blocks, partial_block)

                sublayer_count += 1
                if sublayer_count % self.block_size == 0 and partial_block is not None:
                    blocks.append(partial_block)
                    partial_block = None

                mlp_out = swin_block.drop_path2(swin_block.mlp(swin_block.norm2(h2)))
                partial_block = mlp_out if partial_block is None else partial_block + mlp_out

            pb = partial_block if partial_block is not None else torch.zeros_like(x)
            x = self.attn_res[f"s{si}_final"](blocks, pb)

        x = self.swin.norm(x)
        x = self.swin.head(x)
        return x
