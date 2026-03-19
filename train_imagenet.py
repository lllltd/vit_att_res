"""
CS-AttnRes Training Script
Supports: Swin-T baseline, Swin-T + AttnRes, future CS-AttnRes variants
Tracking: Weights & Biases (wandb)

Usage:
    # ImageNet-100 (development)
    python train_imagenet.py --model baseline --data ~/autodl-tmp/imagenet100 --epochs 100

    # ImageNet-1K (final, on A100)
    python train_imagenet.py --model attnres --data /path/to/imagenet --epochs 300 \
        --bs 256 --num_classes 1000 --warmup 20 --mixup 0.8 --cutmix 1.0

    # Disable wandb
    python train_imagenet.py --model baseline --no_wandb
"""
import os
import sys
import time
import math
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# Optional: wandb
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("wandb not installed. Run: pip install wandb")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import timm
from models.swin_attnres import SwinWithAttnRes


# ──────────────────── Args ────────────────────
def get_args():
    p = argparse.ArgumentParser(description='CS-AttnRes Training')

    # Model
    p.add_argument('--model', default='baseline', choices=['baseline', 'attnres', 'cs_attnres'])
    p.add_argument('--num_classes', default=100, type=int)
    p.add_argument('--drop_path', default=0.2, type=float)

    # Data
    p.add_argument('--data', default=os.path.expanduser('~/autodl-tmp/imagenet100'))
    p.add_argument('--input_size', default=224, type=int)
    p.add_argument('--workers', default=8, type=int)

    # Training
    p.add_argument('--epochs', default=100, type=int)
    p.add_argument('--bs', default=128, type=int, help='batch size per GPU')
    p.add_argument('--lr', default=1e-3, type=float)
    p.add_argument('--min_lr', default=1e-5, type=float)
    p.add_argument('--warmup', default=5, type=int, help='warmup epochs')
    p.add_argument('--wd', default=0.05, type=float, help='weight decay')
    p.add_argument('--grad_clip', default=5.0, type=float)
    p.add_argument('--label_smoothing', default=0.1, type=float)

    # Augmentation
    p.add_argument('--auto_augment', default='imagenet', type=str)
    p.add_argument('--random_erasing', default=0.25, type=float)
    p.add_argument('--mixup', default=0.0, type=float, help='mixup alpha (0=disabled)')
    p.add_argument('--cutmix', default=0.0, type=float, help='cutmix alpha (0=disabled)')

    # Logging
    p.add_argument('--no_wandb', action='store_true')
    p.add_argument('--wandb_project', default='cs-attnres')
    p.add_argument('--wandb_entity', default=None)
    p.add_argument('--seed', default=42, type=int)
    p.add_argument('--output_dir', default='checkpoint')
    p.add_argument('--log_dir', default='logs')
    p.add_argument('--resume', default=None, type=str, help='path to checkpoint to resume')

    return p.parse_args()


# ──────────────────── Data ────────────────────
def build_loader(args):
    train_transforms = [
        transforms.RandomResizedCrop(args.input_size),
        transforms.RandomHorizontalFlip(),
    ]
    if args.auto_augment:
        train_transforms.append(
            transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET))
    train_transforms += [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    if args.random_erasing > 0:
        train_transforms.append(transforms.RandomErasing(p=args.random_erasing))

    val_transforms = [
        transforms.Resize(int(args.input_size * 256 / 224)),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]

    train_ds = datasets.ImageFolder(
        os.path.join(args.data, 'train'), transforms.Compose(train_transforms))
    val_ds = datasets.ImageFolder(
        os.path.join(args.data, 'val'), transforms.Compose(val_transforms))

    train_loader = DataLoader(
        train_ds, batch_size=args.bs, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=args.bs, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader, len(train_ds.classes)


# ──────────────────── Model ────────────────────
def build_model(args):
    if args.model == 'baseline':
        model = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=False,
            num_classes=args.num_classes,
            drop_path_rate=args.drop_path,
        )
    elif args.model == 'attnres':
        model = SwinWithAttnRes(num_classes=args.num_classes)
    elif args.model == 'cs_attnres':
        # TODO: implement Cross-Scale + Spatial-Aware
        raise NotImplementedError("CS-AttnRes coming soon")
    else:
        raise ValueError(f"Unknown model: {args.model}")

    params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    return model, params, trainable


# ──────────────────── Scheduler ────────────────────
def build_scheduler(optimizer, args, steps_per_epoch):
    warmup_steps = args.warmup * steps_per_epoch
    total_steps = args.epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        ratio = args.min_lr / args.lr
        return ratio + (1 - ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ──────────────────── Mixup / CutMix ────────────────────
class MixupCutmix:
    """Simple Mixup + CutMix with random switch."""
    def __init__(self, mixup_alpha=0.8, cutmix_alpha=1.0, num_classes=100):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.num_classes = num_classes
        self.enabled = mixup_alpha > 0 or cutmix_alpha > 0

    def __call__(self, images, targets):
        if not self.enabled:
            return images, targets

        # One-hot targets for mixing
        targets_one_hot = torch.zeros(targets.size(0), self.num_classes, device=targets.device)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1.0)

        # Randomly choose mixup or cutmix
        use_cutmix = (self.cutmix_alpha > 0 and
                      (self.mixup_alpha <= 0 or torch.rand(1).item() > 0.5))

        if use_cutmix:
            lam = torch.distributions.Beta(self.cutmix_alpha, self.cutmix_alpha).sample().item()
        else:
            lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample().item()

        # Shuffle
        idx = torch.randperm(images.size(0), device=images.device)

        if use_cutmix:
            # CutMix: random box
            _, _, H, W = images.shape
            r = math.sqrt(1 - lam)
            rH, rW = int(H * r), int(W * r)
            cH = torch.randint(0, H, (1,)).item()
            cW = torch.randint(0, W, (1,)).item()
            y1, y2 = max(cH - rH // 2, 0), min(cH + rH // 2, H)
            x1, x2 = max(cW - rW // 2, 0), min(cW + rW // 2, W)
            images[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]
            lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)
        else:
            # Mixup
            images = lam * images + (1 - lam) * images[idx]

        targets_mixed = lam * targets_one_hot + (1 - lam) * targets_one_hot[idx]
        return images, targets_mixed


# ──────────────────── Train / Eval ────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler,
                    device, epoch, args, mixup_fn=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    t0 = time.time()

    for i, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Mixup / CutMix
        if mixup_fn is not None and mixup_fn.enabled:
            images, targets_mixed = mixup_fn(images, targets)
            use_soft_targets = True
        else:
            targets_mixed = None
            use_soft_targets = False

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            if use_soft_targets:
                loss = -torch.sum(targets_mixed * torch.log_softmax(outputs, dim=1), dim=1).mean()
            else:
                loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        if not use_soft_targets:
            _, pred = outputs.max(1)
            total += targets.size(0)
            correct += pred.eq(targets).sum().item()
        else:
            _, pred = outputs.max(1)
            total += targets.size(0)
            correct += pred.eq(targets_mixed.argmax(1)).sum().item()

        if (i + 1) % 100 == 0:
            lr = optimizer.param_groups[0]['lr']
            acc = 100. * correct / total if total > 0 else 0
            elapsed = time.time() - t0
            print(f'  [{i+1}/{len(loader)}] loss={total_loss/(i+1):.4f} '
                  f'acc={acc:.1f}% lr={lr:.6f} {elapsed:.0f}s')

    avg_loss = total_loss / len(loader)
    avg_acc = 100. * correct / total if total > 0 else 0
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, targets)

        total_loss += loss.item()
        _, pred = outputs.max(1)
        total += targets.size(0)
        correct += pred.eq(targets).sum().item()

    return total_loss / len(loader), 100. * correct / total


# ──────────────────── Main ────────────────────
def main():
    args = get_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')

    # Print info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\n{'='*60}")
    print(f"  CS-AttnRes | {args.model} | {args.epochs}ep | lr={args.lr} | bs={args.bs}")
    print(f"  GPU: {gpu_name} ({gpu_mem:.0f}GB)")
    print(f"  Data: {args.data}")
    print(f"{'='*60}")

    # Data
    train_loader, val_loader, actual_classes = build_loader(args)
    if actual_classes != args.num_classes:
        print(f"WARNING: Found {actual_classes} classes but --num_classes={args.num_classes}")
        print(f"  Auto-correcting to {actual_classes}")
        args.num_classes = actual_classes

    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Classes: {args.num_classes}")

    # Model
    model, params, trainable = build_model(args)
    model = model.to(device)
    print(f"Model: {args.model} | Total: {params:.2f}M | Trainable: {trainable:.2f}M")

    # Training setup
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = build_scheduler(optimizer, args, len(train_loader))
    scaler = torch.amp.GradScaler('cuda')

    # Mixup / CutMix
    mixup_fn = MixupCutmix(
        mixup_alpha=args.mixup,
        cutmix_alpha=args.cutmix,
        num_classes=args.num_classes,
    ) if args.mixup > 0 or args.cutmix > 0 else None

    # Resume
    start_epoch = 1
    best_acc = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_acc = ckpt.get('best_acc', 0)
        print(f"Resumed from epoch {start_epoch - 1}, best_acc={best_acc:.2f}%")

    # Wandb
    use_wandb = HAS_WANDB and not args.no_wandb
    if use_wandb:
        run_name = f"{args.model}_{os.path.basename(args.data)}_ep{args.epochs}_lr{args.lr}_bs{args.bs}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
        )
        wandb.watch(model, log='gradients', log_freq=500)

    # Dirs
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # CSV log
    dataset_name = os.path.basename(args.data.rstrip('/'))
    csv_path = os.path.join(args.log_dir, f'{args.model}_{dataset_name}.csv')
    log_lines = ['epoch,train_loss,train_acc,val_loss,val_acc,lr,time']

    # ──────────── Training loop ────────────
    print(f"\nStarting training from epoch {start_epoch}...\n")
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler,
            device, epoch, args, mixup_fn)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)

        dt = time.time() - t0
        lr = optimizer.param_groups[0]['lr']
        is_best = va_acc > best_acc

        if is_best:
            best_acc = va_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'args': vars(args),
            }, os.path.join(args.output_dir, f'{args.model}_{dataset_name}_best.pth'))

        # Print
        best_mark = ' *BEST*' if is_best else ''
        print(f'Ep {epoch}/{args.epochs} ({dt:.0f}s) | '
              f'Tr {tr_loss:.4f}/{tr_acc:.1f}% | '
              f'Va {va_loss:.4f}/{va_acc:.1f}%{best_mark}')

        # Log
        log_lines.append(f'{epoch},{tr_loss:.4f},{tr_acc:.2f},{va_loss:.4f},{va_acc:.2f},{lr:.6f},{dt:.0f}')

        if use_wandb:
            wandb.log({
                'epoch': epoch,
                'train/loss': tr_loss,
                'train/acc': tr_acc,
                'val/loss': va_loss,
                'val/acc': va_acc,
                'lr': lr,
                'epoch_time': dt,
                'best_val_acc': best_acc,
            })

    # Save CSV
    with open(csv_path, 'w') as f:
        f.write('\n'.join(log_lines))

    if use_wandb:
        wandb.finish()

    print(f'\n{"="*60}')
    print(f'  Done! Best val acc: {best_acc:.2f}%')
    print(f'  Model saved: {args.output_dir}/{args.model}_{dataset_name}_best.pth')
    print(f'  Log saved: {csv_path}')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
