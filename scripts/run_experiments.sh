#!/bin/bash
# ============================================================
# CS-AttnRes: Experiment Runner
# 
# Task assignment:
#   RTX 4090 (24GB): ImageNet-100 experiments (development)
#   A100 (40/80GB):  ImageNet-1K experiments (final results)
#
# Usage:
#   bash scripts/run_experiments.sh imagenet100   # dev experiments
#   bash scripts/run_experiments.sh imagenet1k    # final experiments
# ============================================================

set -e
DATASET=${1:-imagenet100}

if [ "$DATASET" = "imagenet100" ]; then
    # ── ImageNet-100 (RTX 4090) ──
    DATA=~/autodl-tmp/imagenet100
    EPOCHS=100
    BS=128
    CLASSES=100
    echo "Running ImageNet-100 experiments (RTX 4090)"

elif [ "$DATASET" = "imagenet1k" ]; then
    # ── ImageNet-1K (A100) ──
    DATA=~/autodl-tmp/imagenet
    EPOCHS=300
    BS=256   # A100 80GB can handle 256; use 128 for A100 40GB
    CLASSES=1000
    echo "Running ImageNet-1K experiments (A100)"

else
    echo "Usage: $0 {imagenet100|imagenet1k}"
    exit 1
fi

# ── Experiment 1: Baseline ──
echo ""
echo "=== E1: Baseline ==="
python train_imagenet.py \
    --model baseline \
    --data $DATA \
    --num_classes $CLASSES \
    --epochs $EPOCHS \
    --bs $BS \
    --lr 1e-3 \
    --warmup 5 \
    --wandb_project cs-attnres

# ── Experiment 2: Intra-Stage AttnRes ──
echo ""
echo "=== E2: Intra-Stage AttnRes ==="
python train_imagenet.py \
    --model attnres \
    --data $DATA \
    --num_classes $CLASSES \
    --epochs $EPOCHS \
    --bs $BS \
    --lr 1e-3 \
    --warmup 5 \
    --wandb_project cs-attnres

# ── Experiment 3: Full CS-AttnRes (after implementation) ──
# echo ""
# echo "=== E3: CS-AttnRes ==="
# python train_imagenet.py \
#     --model cs_attnres \
#     --data $DATA \
#     --num_classes $CLASSES \
#     --epochs $EPOCHS \
#     --bs $BS \
#     --lr 1e-3 \
#     --warmup 5 \
#     --wandb_project cs-attnres

echo ""
echo "=== All experiments done ==="
