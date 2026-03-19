# CS-AttnRes: Cross-Scale Spatial-Aware Attention Residuals for Vision Transformers

> Extending [Attention Residuals](https://arxiv.org/abs/2603.15031) to hierarchical Vision Transformers with cross-scale depth attention and spatial-aware dynamic queries.

**Target venue**: NeurIPS 2026 &nbsp;|&nbsp; **Status**: 🔬 Active development

---

## Overview

Standard residual connections accumulate layer outputs with fixed unit weights, causing PreNorm dilution in deep networks. [AttnRes](https://github.com/MoonshotAI/Attention-Residuals) (Kimi Team, 2026) replaces this with learned softmax attention over depth — but only validates on language models.

We extend AttnRes to hierarchical Vision Transformers (Swin, PVT) with three contributions:

| # | Contribution | Description |
|---|---|---|
| **C1** | Cross-Scale AttnRes | Lightweight projection to align representations across stages with different dimensions |
| **C2** | Spatial-Aware Depth Attention | DWConv + bottleneck MLP generates per-token dynamic queries (vision-specific) |
| **C3** | Systematic Analysis | Visualizing depth attention patterns across scales, spatial positions, and object sizes |

## Project Structure

```
CS-AttnRes/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── models/
│   ├── __init__.py
│   ├── attnres.py              # Core: BlockAttnRes, RMSNorm
│   └── swin_attnres.py         # SwinWithAttnRes (timm-based)
│
├── configs/
│   └── default.yaml            # All hyperparameters
│
├── train_imagenet.py           # Main training script (wandb enabled)
├── scripts/
│   ├── prepare_imagenet100.py  # Data preparation
│   └── run_experiments.sh      # Launch all experiments
│
├── checkpoint/                 # Saved model weights (gitignored)
├── logs/                       # Training logs (gitignored)
└── docs/
    └── PROJECT_SUMMARY.md      # Detailed project documentation
```

## Quick Start

### 1. Environment Setup

```bash
# Clone
git clone https://github.com/<your-username>/CS-AttnRes.git
cd CS-AttnRes

# Install dependencies
pip install -r requirements.txt

# Login to wandb (for experiment tracking)
wandb login
```

### 2. Data Preparation

**ImageNet-100** (recommended for development):
```bash
# AutoDL: use public dataset
unzip /autodl-pub/data/ImageNet100/imagenet100.zip -d ~/autodl-tmp/imagenet100
python scripts/prepare_imagenet100.py --src ~/autodl-tmp/imagenet100/imagenet100 --dst ~/autodl-tmp/imagenet100

# Custom: organize into train/ and val/ with ImageFolder structure
```

**ImageNet-1K** (for final results):
```bash
# AutoDL
python /root/autodl-pub/ImageNet/extract_imagenet.py /root/autodl-pub/ImageNet/ILSVRC2012 ~/autodl-tmp/imagenet
```

### 3. Training

```bash
# Baseline (Swin-T, no AttnRes)
python train_imagenet.py --model baseline --data ~/autodl-tmp/imagenet100 --epochs 100

# Intra-Stage AttnRes
python train_imagenet.py --model attnres --data ~/autodl-tmp/imagenet100 --epochs 100

# Full CS-AttnRes (after C1 + C2 are implemented)
python train_imagenet.py --model cs_attnres --data ~/autodl-tmp/imagenet100 --epochs 100
```

All runs are tracked on [Weights & Biases](https://wandb.ai). Set `--no_wandb` to disable.

### 4. View Results

```bash
# Local logs
cat logs/<model>_<dataset>.csv

# wandb dashboard
# https://wandb.ai/<your-entity>/cs-attnres
```

## Current Progress

| Experiment | Dataset | Model | Status | Result |
|---|---|---|---|---|
| Baseline | ImageNet-100 | Swin-T | 🔄 Running | Ep2: 20.5% (in progress) |
| Intra-AttnRes | ImageNet-100 | Swin-T + AttnRes | ⏸️ Paused | Ep3 diverged at bs=64; need to rerun with bs=128 |
| Cross-Scale | ImageNet-100 | Swin-T + CS-AttnRes | ⬜ TODO | - |
| Spatial-Aware | ImageNet-100 | Swin-T + full | ⬜ TODO | - |
| Ablation | ImageNet-100 | Various configs | ⬜ TODO | - |
| Main results | ImageNet-1K | Swin-T/S, PVT-S | ⬜ TODO | - |
| Detection | COCO | Mask R-CNN + Swin-T | ⬜ TODO | - |
| Segmentation | ADE20K | Semantic FPN | ⬜ TODO | - |

### Known Issues

- **AttnRes training divergence**: When trained with bs=64 (half of baseline's bs=128), loss rebounds at epoch 3. Likely caused by inconsistent effective learning rate. **Fix**: always use same bs as baseline (128).

## Task Assignment

| Task | Assignee | Device | Priority |
|---|---|---|---|
| ImageNet-100 baseline + AttnRes (bs=128) | You | RTX 4090 | P0 |
| ImageNet-100 ablation (after AttnRes verified) | You | RTX 4090 | P0 |
| ImageNet-1K Swin-T baseline + AttnRes | Collaborator | A100 | P1 |
| ImageNet-1K Swin-S baseline + AttnRes | Collaborator | A100 | P1 |
| ImageNet-1K PVT-S baseline + AttnRes | Collaborator | A100 | P1 |
| COCO detection (mmdetection) | TBD | A100 | P2 |
| ADE20K segmentation (mmsegmentation) | TBD | A100 | P2 |

## Experiment Configs

All hyperparameters follow DeiT/Swin conventions:

| Param | ImageNet-100 | ImageNet-1K |
|---|---|---|
| Epochs | 100 | 300 |
| Batch size | 128 | 1024 (adjust per GPU) |
| Optimizer | AdamW | AdamW |
| Learning rate | 1e-3 | 1e-3 |
| Min LR | 1e-5 | 1e-5 |
| Weight decay | 0.05 | 0.05 |
| Warmup epochs | 5 | 20 |
| Label smoothing | 0.1 | 0.1 |
| Grad clip | 5.0 | 5.0 |
| Augmentation | AutoAugment + RandomErasing | RandAugment + Mixup + CutMix + RandomErasing |
| Input size | 224×224 | 224×224 |

## Citation

```bibtex
@misc{cs-attnres2026,
  title={CS-AttnRes: Cross-Scale Spatial-Aware Attention Residuals for Hierarchical Vision Transformers},
  author={<authors>},
  year={2026},
}
```

### Based on

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
