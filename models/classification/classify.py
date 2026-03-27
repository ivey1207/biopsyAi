import argparse
from pathlib import Path

import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
import timm


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model_name = ckpt.get("model_name", "efficientnetv2_rw_s")
    model = timm.create_model(model_name, pretrained=False, num_classes=12)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    img_size = ckpt.get("img_size", 224)
    return model, img_size


def main():
    parser = argparse.ArgumentParser(description="Inference-only classification script.")
    parser.add_argument("--images-dir", required=True, type=str)
    parser.add_argument("--model-path", required=True, type=str)
    parser.add_argument("--output-xlsx", required=True, type=str)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, img_size = load_model(Path(args.model_path), device)
    tfm = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    image_paths = sorted(Path(args.images_dir).glob("*.png"), key=lambda p: int(p.stem))
    rows = []
    with torch.no_grad():
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            x = tfm(img).unsqueeze(0).to(device)
            pred = int(model(x).argmax(dim=1).item())
            rows.append({"Image_ID": int(p.stem), "Label": pred})

    out_path = Path(args.output_xlsx)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_excel(out_path, index=False)
    print(f"saved: {out_path} rows={len(rows)}")


if __name__ == "__main__":
    main()
