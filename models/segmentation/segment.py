import argparse
from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch


def postprocess_mask(mask: np.ndarray, min_area: int = 80, max_components: int = 3):
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    clean = np.zeros_like(mask, dtype=np.uint8)

    components = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            components.append((i, area))

    components.sort(key=lambda x: x[1], reverse=True)
    for i, _ in components[: max_components if max_components > 0 else len(components)]:
        clean[labels == i] = 255
    return clean


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    encoder_name = ckpt.get("encoder_name", "efficientnet-b3")
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt.get("img_size", 256), ckpt.get("best_threshold", 0.5)


def main():
    parser = argparse.ArgumentParser(description="Inference-only segmentation script.")
    parser.add_argument("--images-dir", required=True, type=str)
    parser.add_argument("--model-path", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--threshold", default=-1.0, type=float)
    parser.add_argument("--threshold-mode", default="fixed", choices=["fixed", "otsu", "percentile"], type=str)
    parser.add_argument("--percentile", default=85.0, type=float)
    parser.add_argument("--min-area", default=80, type=int)
    parser.add_argument("--max-components", default=3, type=int)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    model, img_size, ckpt_thr = load_model(Path(args.model_path), device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        [*Path(args.images_dir).glob("*.jpg"), *Path(args.images_dir).glob("*.png"), *Path(args.images_dir).glob("*.jpeg")],
        key=lambda p: int("".join(ch for ch in p.stem if ch.isdigit()) or 0),
    )
    with torch.no_grad():
        for p in image_paths:
            image = cv2.imread(str(p), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            x = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            x = np.transpose(x, (2, 0, 1))
            x = torch.tensor(x).unsqueeze(0).to(device)

            logits = model(x)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
            prob_resized = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)

            if args.threshold_mode == "otsu":
                prob_u8 = np.clip(prob_resized * 255.0, 0, 255).astype(np.uint8)
                otsu_thr_u8, _ = cv2.threshold(prob_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                thr = float(otsu_thr_u8) / 255.0
            elif args.threshold_mode == "percentile":
                thr = float(np.percentile(prob_resized, np.clip(args.percentile, 1.0, 99.0)))
            else:
                thr = float(args.threshold if args.threshold >= 0 else ckpt_thr)

            mask = (prob_resized > thr).astype(np.uint8) * 255
            mask = postprocess_mask(mask, min_area=args.min_area, max_components=args.max_components)
            cv2.imwrite(str(out_dir / f"{p.stem}.png"), mask)

    print(f"saved masks to: {out_dir} count={len(image_paths)}")


if __name__ == "__main__":
    main()
