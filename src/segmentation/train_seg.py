import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from torch.optim.lr_scheduler import CosineAnnealingLR


class SegDataset(Dataset):
    def __init__(self, img_dir: Path, mask_dir: Path, size: int = 256, transform=None):
        self.size = size
        self.items = sorted([p for p in img_dir.glob("*.jpg")])
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path = self.items[idx]
        stem = img_path.stem
        mask_path = self.mask_dir / f"{stem}.png"

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]
        else:
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        mask = (mask > 127).astype(np.float32)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        mask = np.expand_dims(mask, axis=0)
        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)


def iou_score(logits, targets, thr=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > thr).float()
    inter = (preds * targets).sum(dim=(1, 2, 3))
    union = (preds + targets - preds * targets).sum(dim=(1, 2, 3)) + 1e-7
    return (inter / union).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-images", default="data/Segmentation/training/images", type=str)
    parser.add_argument("--train-masks", default="data/Segmentation/training/masks", type=str)
    parser.add_argument("--val-images", default="data/Segmentation/validation/images", type=str)
    parser.add_argument("--val-masks", default="data/Segmentation/validation/masks", type=str)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--img-size", default=256, type=int)
    parser.add_argument("--encoder-name", default="efficientnet-b3", type=str)
    parser.add_argument("--out", default="models/segmentation/segmenter.pt", type=str)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Using device: {device}")
    
    torch.manual_seed(42)
    np.random.seed(42)

    train_transform = A.Compose([
        A.Resize(args.img_size, args.img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
            A.GridDistoration(p=0.5),
            A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1)
        ], p=0.3),
        A.RandomBrightnessContrast(p=0.3),
    ])

    val_transform = A.Compose([
        A.Resize(args.img_size, args.img_size),
    ])

    train_ds = SegDataset(Path(args.train_images), Path(args.train_masks), args.img_size, transform=train_transform)
    val_ds = SegDataset(Path(args.val_images), Path(args.val_masks), args.img_size, transform=val_transform)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=min(os.cpu_count(), 4), pin_memory=True if torch.cuda.is_available() else False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=min(os.cpu_count(), 4), pin_memory=True if torch.cuda.is_available() else False)

    model = smp.Unet(
        encoder_name=args.encoder_name,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(device)
    
    bce = torch.nn.BCEWithLogitsLoss()
    dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_iou = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = 0.5 * bce(logits, y) + 0.5 * dice(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        
        train_loss /= len(train_loader.dataset)
        scheduler.step()

        model.eval()
        val_iou = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_iou += iou_score(logits, y).item() * x.size(0)
        
        val_iou /= len(val_loader.dataset)
        print(f"Epoch {epoch}/{args.epochs} | Loss: {train_loss:.4f} | Val IoU: {val_iou:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        if val_iou >= best_iou:
            best_iou = val_iou
            torch.save(
                {
                    "img_size": args.img_size,
                    "state_dict": model.state_dict(),
                    "encoder_name": args.encoder_name,
                },
                out_path,
            )
            print(f"--> Saved best checkpoint (iou={best_iou:.4f})")

    print(f"\nTraining Complete. Best Validation IoU: {best_iou:.4f}")


if __name__ == "__main__":
    main()

