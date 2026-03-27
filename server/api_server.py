import base64
import io
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import timm
import segmentation_models_pytorch as smp
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _b64_png_from_numpy(arr: np.ndarray) -> str:
    """
    Convert uint8 numpy array to base64-encoded PNG.
    arr: HxW (grayscale) or HxWx3 (RGB)
    """
    if arr.dtype != np.uint8:
        raise ValueError("Expected uint8 array for PNG encoding.")
    im = Image.fromarray(arr)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def postprocess_mask(mask: np.ndarray, min_area: int = 80, max_components: int = 3) -> np.ndarray:
    """
    Binary mask cleanup:
    - morphology open/close
    - keep largest connected components by area
    """
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
    topk = max_components if max_components > 0 else len(components)
    for i, _ in components[:topk]:
        clean[labels == i] = 255
    return clean


def compute_reference_hist(data_dir: str = "data/classification/train", max_samples: int = 700) -> Optional[np.ndarray]:
    root = (PROJECT_ROOT / data_dir).resolve()
    if not root.exists():
        return None
    paths = sorted(root.glob("*/*.png"))[:max_samples]
    if not paths:
        return None

    acc = None
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256]).astype(np.float32)
        hist /= (hist.sum() + 1e-7)
        acc = hist if acc is None else (acc + hist)

    if acc is None:
        return None
    acc /= (np.sum(acc) + 1e-7)
    return acc


def is_non_biopsy(image: Image.Image, ref_hist: Optional[np.ndarray], hist_corr_threshold: float = 0.25) -> Tuple[bool, str]:
    arr = np.array(image.convert("RGB"))
    h, w = arr.shape[:2]
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Strong cue: frontal face detected -> out-of-domain for biopsy task.
    face_model = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
    if len(faces) > 0:
        return True, "Face-like content detected (out-of-domain)."

    if ref_hist is None:
        return False, "Reference biopsy histogram unavailable."

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256]).astype(np.float32)
    hist /= (hist.sum() + 1e-7)
    corr = float(cv2.compareHist(ref_hist.flatten(), hist.flatten(), cv2.HISTCMP_CORREL))

    # Additional rough sanity checks for generic photos.
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    aspect = max(h, w) / max(1.0, min(h, w))
    if corr < hist_corr_threshold and (aspect > 1.6 or lap_var < 40):
        return True, f"Low biopsy similarity (hist-corr={corr:.2f})."

    return False, f"Biopsy similarity score={corr:.2f}"


def predict_class(image: Image.Image, model: torch.nn.Module, img_size: int, device: torch.device) -> Tuple[int, float]:
    tfm = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    x = tfm(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    pred = int(np.argmax(probs))
    conf = float(probs[pred])
    return pred, conf


def predict_mask(
    image: Image.Image,
    model: torch.nn.Module,
    img_size: int,
    threshold_mode: str = "fixed",
    threshold: float = 0.5,
    percentile: float = 85.0,
    min_area: int = 80,
    max_components: int = 3,
    device: torch.device = torch.device("cpu"),
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    arr = np.array(image.convert("RGB"))
    h, w = arr.shape[:2]

    x = cv2.resize(arr, (img_size, img_size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    x = torch.tensor(x).unsqueeze(0).to(device)

    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0, 0].detach().cpu().numpy()
    prob_resized = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)

    if threshold_mode == "otsu":
        prob_u8 = np.clip(prob_resized * 255.0, 0, 255).astype(np.uint8)
        otsu_thr_u8, _ = cv2.threshold(prob_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = float(otsu_thr_u8) / 255.0
    elif threshold_mode == "percentile":
        thr = float(np.percentile(prob_resized, np.clip(percentile, 1.0, 99.0)))
    else:
        thr = float(threshold)

    mask = (prob_resized > thr).astype(np.uint8) * 255
    mask = postprocess_mask(mask, min_area=min_area, max_components=max_components)
    coverage = float((mask > 0).mean())

    # If mask is empty, auto-relax threshold and area filtering.
    if coverage < 0.0005:
        mask2 = (prob_resized > max(0.35, thr - 0.2)).astype(np.uint8) * 255
        mask2 = postprocess_mask(
            mask2,
            min_area=max(20, min_area // 3),
            max_components=max_components,
        )
        coverage2 = float((mask2 > 0).mean())
        if coverage2 > coverage:
            mask = mask2
            coverage = coverage2

    return mask, prob_resized, coverage, thr


def _distance_map_png(mask: np.ndarray) -> str:
    """Create distance map PNG (b64) from binary mask with values 0/255."""
    bin_u8 = (mask > 0).astype(np.uint8)
    dist = cv2.distanceTransform(bin_u8, cv2.DIST_L2, 3)
    if np.max(dist) > 0:
        dist_u8 = (dist / np.max(dist) * 255.0).astype(np.uint8)
    else:
        dist_u8 = np.zeros_like(mask, dtype=np.uint8)
    return _b64_png_from_numpy(dist_u8)


def _contour_overlay_png(image_rgb: np.ndarray, mask: np.ndarray) -> str:
    """Draw external contours on top of the original RGB image (b64 PNG)."""
    mask_u8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = image_rgb.copy()
    cv2.drawContours(overlay, contours, -1, (50, 255, 120), 2)
    return _b64_png_from_numpy(overlay.astype(np.uint8))


def _components_from_mask(mask: np.ndarray, image_rgb: np.ndarray, top_k: int) -> list[dict]:
    """
    Split cleaned mask into connected components and return top-k by area.

    Each component includes:
    - name
    - area_pixels
    - mask_png_b64
    - overlay_png_b64
    """
    mask_u8 = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    comps: list[tuple[int, int]] = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area > 0:
            comps.append((i, area))

    comps.sort(key=lambda x: x[1], reverse=True)
    if top_k <= 0:
        top_k = len(comps)

    out: list[dict] = []
    for idx, (label_id, area) in enumerate(comps[:top_k], start=1):
        comp_mask = (labels == label_id).astype(np.uint8) * 255
        comp_overlay = image_rgb.copy()
        comp_overlay[comp_mask > 0] = [255, 80, 80]
        comp_overlay = (
            image_rgb.astype(np.float32) * 0.65 + comp_overlay.astype(np.float32) * 0.35
        ).clip(0, 255).astype(np.uint8)

        out.append(
            {
                "name": f"Очаг #{idx}",
                "area_pixels": area,
                "mask_png_b64": _b64_png_from_numpy(comp_mask.astype(np.uint8)),
                "overlay_png_b64": _b64_png_from_numpy(comp_overlay.astype(np.uint8)),
            }
        )
    return out


def load_class_names_ru() -> Dict[int, str]:
    # User can update mapping for 0..11 in config/class_names.json
    config_path = PROJECT_ROOT / "config" / "class_names.json"
    fallback = {i: f"Class {i}" for i in range(12)}
    if not config_path.exists():
        return fallback
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        out: Dict[int, str] = {}
        for k, v in data.items():
            idx = int(k)
            if isinstance(v, dict) and "ru" in v:
                out[idx] = str(v["ru"])
            elif isinstance(v, str):
                out[idx] = v
        # Ensure all 0..11 exist
        for i in range(12):
            out.setdefault(i, fallback[i])
        return out
    except Exception:
        return fallback


app = FastAPI(title="Biopsy AI - API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REF_HIST: Optional[np.ndarray] = None
CLS_MODEL: Optional[torch.nn.Module] = None
SEG_MODEL: Optional[torch.nn.Module] = None
CLS_IMG_SIZE: int = 224
SEG_IMG_SIZE: int = 256
CLASS_NAMES_RU: Dict[int, str] = load_class_names_ru()


@app.on_event("startup")
def _startup() -> None:
    global REF_HIST, CLS_MODEL, SEG_MODEL, CLS_IMG_SIZE, SEG_IMG_SIZE

    # Histogram reference is optional: if dataset isn't shipped, domain check stays off.
    REF_HIST = compute_reference_hist()

    cls_path = PROJECT_ROOT / "models" / "classification" / "classifier.pt"
    seg_path = PROJECT_ROOT / "models" / "segmentation" / "segmenter.pt"

    if cls_path.exists():
        ckpt = torch.load(cls_path, map_location="cpu")
        model_name = ckpt.get("model_name", "efficientnetv2_rw_s")
        classes_val = ckpt.get("classes", 12)
        if isinstance(classes_val, (list, tuple)):
            num_classes = len(classes_val)
        else:
            num_classes = int(classes_val)
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        model.load_state_dict(ckpt["state_dict"])
        model.to(DEVICE)
        model.eval()
        CLS_MODEL = model
        CLS_IMG_SIZE = int(ckpt.get("img_size", 224))

    if seg_path.exists():
        ckpt = torch.load(seg_path, map_location="cpu")
        encoder_name = ckpt.get("encoder_name", "efficientnet-b3")
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )
        model.load_state_dict(ckpt["state_dict"])
        model.to(DEVICE)
        model.eval()
        SEG_MODEL = model
        SEG_IMG_SIZE = int(ckpt.get("img_size", 256))


def _run_inference(
    image: Image.Image,
    threshold_mode: str,
    threshold: float,
    percentile: float,
    min_area: int,
    max_components: int,
    run_seg: bool,
) -> Dict[str, Any]:
    non_biopsy, reason = is_non_biopsy(image, REF_HIST)
    result: Dict[str, Any] = {
        "non_biopsy": non_biopsy,
        "reason": reason,
    }

    if non_biopsy or CLS_MODEL is None:
        result["skip_seg"] = True
        return result

    pred, conf = predict_class(image, CLS_MODEL, CLS_IMG_SIZE, DEVICE)
    result["pred_class"] = pred
    result["pred_class_name_ru"] = CLASS_NAMES_RU.get(pred, f"Class {pred}")
    result["confidence"] = conf

    if not run_seg or SEG_MODEL is None:
        result["segmentation"] = None
        result["skip_seg"] = True
        return result

    mask, prob_map, coverage, used_thr = predict_mask(
        image=image,
        model=SEG_MODEL,
        img_size=SEG_IMG_SIZE,
        threshold_mode=threshold_mode,
        threshold=threshold,
        percentile=percentile,
        min_area=min_area,
        max_components=max_components,
        device=DEVICE,
    )

    # Overlay: mask red where mask > 0
    rgb = np.array(image.convert("RGB"))
    overlay = rgb.copy()
    overlay[mask > 0] = [255, 80, 80]
    blend = (rgb.astype(np.float32) * 0.65 + overlay.astype(np.float32) * 0.35).clip(0, 255).astype(np.uint8)

    components = _components_from_mask(mask=mask.astype(np.uint8), image_rgb=rgb.astype(np.uint8), top_k=max_components)
    # Generate VARIANTS for comparison view
    used_thr = float(used_thr)
    
    def get_mask_b64(m):
        return _b64_png_from_numpy(m.astype(np.uint8))

    # Variant 1: Classical Otsu (Traditional ML)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, otsu_mask_raw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_mask = 255 - otsu_mask_raw # Invert usually needed for medical light background

    # Variant 2: High Sensitivity (Low Threshold)
    high_sens_mask = (prob_map > 0.3).astype(np.uint8) * 255
    
    # Variant 3: Selective (High Threshold)
    selective_mask = (prob_map > 0.7).astype(np.uint8) * 255
    
    # Variant 4: Morphologically Cleaned
    kernel = np.ones((5,5), np.uint8)
    curated_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    seg_obj = {
        "maps": {
            "mask": {"png_b64": get_mask_b64(mask)},
            "overlay": {"png_b64": get_mask_b64(blend)},
            "prob": {"png_b64": _b64_png_from_numpy((np.clip(prob_map, 0, 1) * 255.0).astype(np.uint8))},
            "distance": {"png_b64": _distance_map_png(mask.astype(np.uint8))},
            "contour": {"png_b64": _contour_overlay_png(rgb.astype(np.uint8), mask.astype(np.uint8))},
        },
        "variants": [
            {"name": "Neural Network (Your Model)", "png_b64": get_mask_b64(mask), "type": "ai"},
            {"name": "Classical Otsu", "png_b64": get_mask_b64(otsu_mask), "type": "legacy"},
            {"name": "High Sensitivity (0.3)", "png_b64": get_mask_b64(high_sens_mask), "type": "variant"},
            {"name": "Selective Model (0.7)", "png_b64": get_mask_b64(selective_mask), "type": "variant"},
            {"name": "Curated Polish", "png_b64": get_mask_b64(curated_mask), "type": "ai"},
        ],
        "components": components,
        "coverage": coverage,
        "thr_used": used_thr,
        "threshold_mode": threshold_mode,
        "num_components": len(components),
    }
    result["segmentation"] = seg_obj
    result["skip_seg"] = False
    return result


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"ok": True})


@app.post("/api/predict")
async def api_predict(
    file: UploadFile = File(...),
    threshold_mode: str = Form("fixed"),
    threshold: float = Form(0.65),
    percentile: float = Form(85.0),
    min_area: int = Form(40),
    max_components: int = Form(3),
    run_seg: bool = Form(True),
) -> JSONResponse:
    if threshold_mode not in {"fixed", "otsu", "percentile"}:
        return JSONResponse(status_code=400, content={"error": "Invalid threshold_mode"})

    raw = await file.read()
    if not raw:
        return JSONResponse(status_code=400, content={"error": "Empty file"})

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Unsupported image"})

    out = await run_in_threadpool(
        _run_inference,
        image,
        threshold_mode,
        _safe_float(threshold),
        _safe_float(percentile),
        int(min_area),
        int(max_components),
        bool(run_seg),
    )
    return JSONResponse(out)


FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")

