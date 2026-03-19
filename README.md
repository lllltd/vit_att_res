# CS-AttnRes

> Cross-Scale Spatial-Aware Attention Residuals for Hierarchical Vision Transformers

Extending [Attention Residuals](https://arxiv.org/abs/2603.15031) (Kimi Team, 2026) to hierarchical Vision Transformers with cross-scale depth attention and spatial-aware dynamic queries.

## Quick Start

```bash
git clone https://github.com/<your-username>/CS-AttnRes.git
cd CS-AttnRes
pip install -r requirements.txt
wandb login  # optional, for experiment tracking
```

## Supported Models

| Model | Arch flag | Params | Layers | Stage depths |
|-------|-----------|--------|--------|-------------|
| Swin-T | `swin_tiny` | 28M | 12 | (2,2,6,2) |
| Swin-S | `swin_small` | 49M | 24 | (2,2,**18**,2) |
| Swin-B | `swin_base` | 88M | 24 | (2,2,**18**,2) |

**Note**: AttnRes benefits most from deeper models. Swin-S/B with 18-layer Stage 3 is the primary target.

## Training

```bash
# Swin-S baseline on ImageNet-100 (RTX 4090, ~5h)
python train_imagenet.py --model baseline --arch swin_small \
    --data ~/autodl-tmp/imagenet100 --epochs 100 --bs 128

# Swin-S + AttnRes on ImageNet-100 (RTX 4090, ~5h)
python train_imagenet.py --model attnres --arch swin_small \
    --data ~/autodl-tmp/imagenet100 --epochs 100 --bs 128

# Swin-T baseline on ImageNet-1K (V100/A100, ~5 days)
python train_imagenet.py --model baseline --arch swin_tiny \
    --data ~/autodl-tmp/imagenet --num_classes 1000 --epochs 300 \
    --bs 128 --warmup 20 --mixup 0.8 --cutmix 1.0

# Swin-S + AttnRes on ImageNet-1K (V100/A100, ~7 days)
python train_imagenet.py --model attnres --arch swin_small \
    --data ~/autodl-tmp/imagenet --num_classes 1000 --epochs 300 \
    --bs 128 --warmup 20 --mixup 0.8 --cutmix 1.0
```

## Data Preparation

**ImageNet-100** (development, ~14GB):
```bash
# AutoDL
unzip /autodl-pub/data/ImageNet100/imagenet100.zip -d ~/autodl-tmp/imagenet100
python scripts/prepare_imagenet100.py \
    --src ~/autodl-tmp/imagenet100/imagenet100 \
    --dst ~/autodl-tmp/imagenet100
```

**ImageNet-1K** (final results, ~150GB):
```bash
# AutoDL
python /root/autodl-pub/ImageNet/extract_imagenet.py \
    /root/autodl-pub/ImageNet/ILSVRC2012 ~/autodl-tmp/imagenet
```

## Current Results

| Model | Arch | Dataset | Epochs | Best Val Acc | Status |
|-------|------|---------|--------|-------------|--------|
| Baseline | Swin-T | ImageNet-100 | 100 | **84.25%** | ✅ Done |
| Baseline | Swin-S | ImageNet-100 | 100 | - | 🔄 Running |
| AttnRes | Swin-S | ImageNet-100 | 100 | - | ⬜ Next |
| Baseline | Swin-T | ImageNet-1K | 300 | - | ⬜ Collaborator |
| AttnRes | Swin-T | ImageNet-1K | 300 | - | ⬜ Collaborator |

## Project Structure

```
CS-AttnRes/
├── models/
│   ├── __init__.py
│   ├── attnres.py           # BlockAttnRes core module
│   └── swin_attnres.py      # Swin + AttnRes (supports T/S/B)
├── train_imagenet.py         # Main training script (wandb)
├── scripts/
│   └── prepare_imagenet100.py
├── configs/
│   └── default.yaml
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Hyperparameters

| Param | ImageNet-100 | ImageNet-1K |
|-------|-------------|-------------|
| Epochs | 100 | 300 |
| Batch size | 128 | 128 (V100) / 256 (A100) |
| Optimizer | AdamW | AdamW |
| LR | 1e-3 | 1e-3 |
| Min LR | 1e-5 | 1e-5 |
| Warmup | 5 | 20 |
| Weight decay | 0.05 | 0.05 |
| Label smoothing | 0.1 | 0.1 |
| Mixup | 0 | 0.8 |
| CutMix | 0 | 1.0 |
| Drop path | 0.2 | 0.2 |

## Citation

```bibtex
@misc{chen2026attnres,
  title={Attention Residuals},
  author={Kimi Team},
  year={2026},
  eprint={2603.15031},
  archivePrefix={arXiv},
}
```

## License

MIT
