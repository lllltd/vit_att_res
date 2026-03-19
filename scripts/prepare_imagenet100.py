"""
Prepare ImageNet-100 dataset: split into train/val using symlinks.
Usage:
    python scripts/prepare_imagenet100.py --src ~/autodl-tmp/imagenet100/imagenet100 --dst ~/autodl-tmp/imagenet100
"""
import os
import random
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True, help='Source dir with class folders')
    parser.add_argument('--dst', required=True, help='Destination dir for train/val split')
    parser.add_argument('--val_ratio', default=0.1, type=float)
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    random.seed(args.seed)

    dst_train = os.path.join(args.dst, 'train')
    dst_val = os.path.join(args.dst, 'val')

    classes = sorted([d for d in os.listdir(args.src)
                      if os.path.isdir(os.path.join(args.src, d))])
    print(f"Found {len(classes)} classes")

    for cls in classes:
        cls_dir = os.path.join(args.src, cls)
        imgs = sorted([f for f in os.listdir(cls_dir) if f.endswith('.JPEG') or f.endswith('.jpg') or f.endswith('.png')])
        random.shuffle(imgs)

        split = int(len(imgs) * (1 - args.val_ratio))
        train_imgs = imgs[:split]
        val_imgs = imgs[split:]

        for img in train_imgs:
            dst = os.path.join(dst_train, cls)
            os.makedirs(dst, exist_ok=True)
            link = os.path.join(dst, img)
            if not os.path.exists(link):
                os.symlink(os.path.join(cls_dir, img), link)

        for img in val_imgs:
            dst = os.path.join(dst_val, cls)
            os.makedirs(dst, exist_ok=True)
            link = os.path.join(dst, img)
            if not os.path.exists(link):
                os.symlink(os.path.join(cls_dir, img), link)

    train_total = sum(len(os.listdir(os.path.join(dst_train, c))) for c in os.listdir(dst_train))
    val_total = sum(len(os.listdir(os.path.join(dst_val, c))) for c in os.listdir(dst_val))
    print(f"Train: {train_total} images | Val: {val_total} images")
    print("Done!")


if __name__ == '__main__':
    main()
