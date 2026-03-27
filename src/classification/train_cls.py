import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
from torch.optim.lr_scheduler import CosineAnnealingLR


class ClsDataset(Dataset):
    def __init__(self, items, tfm):
        self.items = items
        self.tfm = tfm

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        image = Image.open(path).convert("RGB")
        return self.tfm(image), label


def build_items(train_root: Path):
    items = []
    # Support both numeric and non-numeric directory names if necessary, 
    # but the task specifies classes 0..11.
    for cls_dir in sorted(train_root.iterdir()):
        if not cls_dir.is_dir() or not cls_dir.name.isdigit():
            continue
        label = int(cls_dir.name)
        for img_path in list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpg")):
            items.append((img_path, label))
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=str, default="data/classification/train")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--model-name", type=str, default="efficientnetv2_s")
    parser.add_argument("--out", type=str, default="models/classification/classifier.pt")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Using device: {device}")
    
    torch.manual_seed(42)
    np.random.seed(42)

    train_root = Path(args.train_dir)
    if not train_root.exists():
        print(f"Error: {train_root} does not exist.")
        return

    items = build_items(train_root)
    if not items:
        print(f"Error: No images found in {train_root}")
        return

    labels = [y for _, y in items]
    train_items, val_items = train_test_split(
        items, test_size=0.15, random_state=42, stratify=labels
    )

    # Improved augmentations
    train_tfm = transforms.Compose(
        [
            transforms.RandomResizedCrop(args.img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    val_tfm = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_loader = DataLoader(
        ClsDataset(train_items, train_tfm),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=min(os.cpu_count(), 4),
        pin_memory=True if torch.cuda.is_available() else False,
    )
    val_loader = DataLoader(
        ClsDataset(val_items, val_tfm),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(os.cpu_count(), 4),
        pin_memory=True if torch.cuda.is_available() else False,
    )

    model = timm.create_model(args.model_name, pretrained=True, num_classes=12)
    model.to(device)
    
    # Class weights for imbalance
    class_counts = np.bincount(np.array(labels), minlength=12).astype(np.float32)
    class_weights = (class_counts.sum() / (len(class_counts) * np.maximum(class_counts, 1.0)))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        
        train_loss /= len(train_loader.dataset)
        scheduler.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        val_acc = correct / max(total, 1)
        print(f"Epoch {epoch}/{args.epochs} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_name": args.model_name,
                    "img_size": args.img_size,
                    "state_dict": model.state_dict(),
                    "classes": list(range(12)),
                },
                out_path,
            )
            print(f"--> Saved best checkpoint (acc={best_acc:.4f})")

    print(f"\nTraining Complete. Best Validation Accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()

