"""
Prepare ImageNet-100: split into train/val using symlinks.
Usage: python scripts/prepare_imagenet100.py --src ~/autodl-tmp/imagenet100/imagenet100 --dst ~/autodl-tmp/imagenet100
"""
import os, random, argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True)
    p.add_argument('--dst', required=True)
    p.add_argument('--val_ratio', default=0.1, type=float)
    p.add_argument('--seed', default=42, type=int)
    args = p.parse_args()

    random.seed(args.seed)
    dst_train = os.path.join(args.dst, 'train')
    dst_val = os.path.join(args.dst, 'val')

    classes = sorted([d for d in os.listdir(args.src) if os.path.isdir(os.path.join(args.src, d))])
    print(f"Found {len(classes)} classes")

    for cls in classes:
        cls_dir = os.path.join(args.src, cls)
        imgs = sorted([f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpeg','.jpg','.png'))])
        random.shuffle(imgs)
        split = int(len(imgs) * (1 - args.val_ratio))
        for img in imgs[:split]:
            d = os.path.join(dst_train, cls); os.makedirs(d, exist_ok=True)
            link = os.path.join(d, img)
            if not os.path.exists(link): os.symlink(os.path.join(cls_dir, img), link)
        for img in imgs[split:]:
            d = os.path.join(dst_val, cls); os.makedirs(d, exist_ok=True)
            link = os.path.join(d, img)
            if not os.path.exists(link): os.symlink(os.path.join(cls_dir, img), link)

    t = sum(len(os.listdir(os.path.join(dst_train, c))) for c in os.listdir(dst_train))
    v = sum(len(os.listdir(os.path.join(dst_val, c))) for c in os.listdir(dst_val))
    print(f"Train: {t} | Val: {v} | Done!")

if __name__ == '__main__':
    main()
