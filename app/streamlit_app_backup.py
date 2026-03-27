from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import streamlit as st
import timm
import torch
from torchvision import transforms
import segmentation_models_pytorch as smp


st.set_page_config(page_title="Biopsy AI Demo", layout="wide")

TEXTS = {
    "ru": {
        "title": "Biopsy AI Assistant",
        "subtitle": "Классификация + сегментация",
        "badge": "Быстрый локальный анализ",
        "disclaimer": "Только для исследования и демонстрации. Не для клинического применения.",
        "upload": "Загрузите изображение биопсии",
        "cls_model": "Модель классификации",
        "seg_model": "Модель сегментации",
        "thr_mode": "Режим порога",
        "thr": "Порог сегментации",
        "perc": "Процентильный порог (%)",
        "min_area": "Минимальная площадь очага (px)",
        "max_comp": "Макс. число компонент",
        "run": "Запустить анализ",
        "results": "Результаты",
        "upload_hint": "Загрузите изображение для старта.",
        "pred_class": "Предсказанный класс",
        "confidence": "уверенность",
        "non_biopsy": "Не-биопсия",
        "rejected": "Отклонено проверкой домена",
        "quality": "Проверка входа",
        "skip_seg": "Сегментация пропущена: вход не похож на биопсию.",
        "mask": "Бинарная маска",
        "overlay": "Наложение маски",
        "prob": "Карта вероятностей",
        "contour": "Карта контуров",
        "dist": "Distance map (карта расстояний)",
        "coverage": "Покрытие маски",
        "thr_used": "использованный порог",
        "model_nf_cls": "Файл модели классификации не найден.",
        "model_nf_seg": "Файл модели сегментации не найден.",
        "guide_title": "Как читать результат",
        "guide_md": (
            "- **Классификация**: число 0..11 — класс заболевания из ТЗ.\n"
            "- **Бинарная маска**: белое = предполагаемая зона интереса.\n"
            "- **Overlay**: маска поверх исходного изображения.\n"
            "- **Probability map**: где модель уверена сильнее.\n"
            "- **Threshold mode**:\n"
            "  - `fixed` — ручной порог.\n"
            "  - `otsu` — авто-порог от изображения.\n"
            "  - `percentile` — сохраняет верхние уверенные пиксели."
        ),
        "ru_help_title": "Описание (Русский)",
        "ru_help_md": (
            "1) Загрузите изображение биопсии.\n"
            "2) Нажмите **Запустить анализ**.\n"
            "3) Для сложных изображений используйте `otsu` или `percentile`."
        ),
        "uz_help_title": "Tavsif (O'zbekcha)",
        "uz_help_md": (
            "1) Biopsiya rasmini yuklang.\n"
            "2) **Tahlilni boshlash** tugmasini bosing.\n"
            "3) Qiyin rasmlar uchun `otsu` yoki `percentile` rejimini tanlang."
        ),
    },
    "uz": {
        "title": "Biopsy AI Assistant",
        "subtitle": "Klassifikatsiya + segmentatsiya",
        "badge": "Tezkor lokal tahlil",
        "disclaimer": "Faqat tadqiqot va demo uchun. Klinik qo'llash uchun emas.",
        "upload": "Biopsiya rasmini yuklang",
        "cls_model": "Klassifikatsiya modeli",
        "seg_model": "Segmentatsiya modeli",
        "thr_mode": "Threshold rejimi",
        "thr": "Segmentatsiya threshold",
        "perc": "Percentile threshold (%)",
        "min_area": "Minimal o'choq maydoni (px)",
        "max_comp": "Maksimal komponentlar soni",
        "run": "Tahlilni boshlash",
        "results": "Natijalar",
        "upload_hint": "Boshlash uchun rasm yuklang.",
        "pred_class": "Bashorat qilingan sinf",
        "confidence": "ishonchlilik",
        "non_biopsy": "Biopsiya emas",
        "rejected": "Domen tekshiruvida rad etildi",
        "quality": "Kirish tekshiruvi",
        "skip_seg": "Segmentatsiya o'tkazib yuborildi: kirish biopsiyaga o'xshamaydi.",
        "mask": "Binar maska",
        "overlay": "Maska overlay",
        "prob": "Ehtimollik xaritasi",
        "contour": "Kontur xaritasi",
        "dist": "Distance map (masofa xaritasi)",
        "coverage": "Maska qamrovi",
        "thr_used": "ishlatilgan threshold",
        "model_nf_cls": "Klassifikatsiya modeli topilmadi.",
        "model_nf_seg": "Segmentatsiya modeli topilmadi.",
        "guide_title": "Natijani tushunish",
        "guide_md": (
            "- **Klassifikatsiya**: 0..11 oralig'idagi sinf.\n"
            "- **Binar maska**: oq hudud — qiziqish zonasi.\n"
            "- **Overlay**: maska + original rasm.\n"
            "- **Probability map**: model qayerda ko'proq ishonadi.\n"
            "- **Threshold mode**:\n"
            "  - `fixed` — qo'lda threshold.\n"
            "  - `otsu` — avtomatik threshold.\n"
            "  - `percentile` — eng ishonchli piksellar."
        ),
        "ru_help_title": "Описание (Русский)",
        "ru_help_md": (
            "1) Загрузите изображение биопсии.\n"
            "2) Нажмите **Запустить анализ**.\n"
            "3) Для сложных изображений используйте `otsu` или `percentile`."
        ),
        "uz_help_title": "Tavsif (O'zbekcha)",
        "uz_help_md": (
            "1) Biopsiya rasmini yuklang.\n"
            "2) **Tahlilni boshlash** tugmasini bosing.\n"
            "3) Qiyin rasmlar uchun `otsu` yoki `percentile` rejimini tanlang."
        ),
    },
    "en": {
        "title": "Biopsy AI Assistant",
        "subtitle": "Classification + Segmentation",
        "badge": "Fast local inference",
        "disclaimer": "For research and demonstration purposes only. Not for clinical use.",
        "upload": "Upload biopsy image",
        "cls_model": "Classification model",
        "seg_model": "Segmentation model",
        "thr_mode": "Threshold mode",
        "thr": "Segmentation threshold",
        "perc": "Percentile threshold (%)",
        "min_area": "Min lesion area (px)",
        "max_comp": "Max connected components",
        "run": "Run analysis",
        "results": "Results",
        "upload_hint": "Upload image to start.",
        "pred_class": "Predicted class",
        "confidence": "confidence",
        "non_biopsy": "Non-biopsy",
        "rejected": "Rejected by domain check",
        "quality": "Input quality check",
        "skip_seg": "Segmentation skipped: input is not biopsy-like.",
        "mask": "Predicted binary mask",
        "overlay": "Segmentation overlay",
        "prob": "Probability map",
        "contour": "Contour map",
        "dist": "Distance map",
        "coverage": "Mask coverage",
        "thr_used": "threshold used",
        "model_nf_cls": "Classification model not found.",
        "model_nf_seg": "Segmentation model not found.",
        "guide_title": "How to read output",
        "guide_md": (
            "- **Classification**: class id 0..11 from challenge labels.\n"
            "- **Binary mask**: white = predicted region of interest.\n"
            "- **Overlay**: mask on top of the input image.\n"
            "- **Probability map**: where model is more confident.\n"
            "- **Threshold mode**:\n"
            "  - `fixed` — manual threshold.\n"
            "  - `otsu` — image-adaptive automatic threshold.\n"
            "  - `percentile` — keeps top confident pixels."
        ),
        "ru_help_title": "Описание (Русский)",
        "ru_help_md": (
            "1) Загрузите изображение биопсии.\n"
            "2) Нажмите **Запустить анализ**.\n"
            "3) Для сложных изображений используйте `otsu` или `percentile`."
        ),
        "uz_help_title": "Tavsif (O'zbekcha)",
        "uz_help_md": (
            "1) Biopsiya rasmini yuklang.\n"
            "2) **Tahlilni boshlash** tugmasini bosing.\n"
            "3) Qiyin rasmlar uchun `otsu` yoki `percentile` rejimini tanlang."
        ),
    },
}

st.markdown(
    """
    <style>
    .stApp { background: radial-gradient(circle at 18% 14%, #1d2a44 0%, #0e1117 40%, #090b10 100%); color: #f5f5f5; }
    .block {
      background: rgba(15,18,26,0.90); border: 1px solid #273041; border-radius: 18px; padding: 18px;
      box-shadow: 0 16px 40px rgba(0,0,0,0.35);
    }
    .title { text-align: center; font-size: 38px; margin-top: 8px; margin-bottom: 0; font-weight: 700; letter-spacing: 0.4px; }
    .subtitle { text-align: center; color: #9aa8c7; margin-bottom: 20px; }
    .badge { display:inline-block; padding:6px 10px; border:1px solid #2c3648; border-radius:999px; font-size:12px; color:#a7b6d8; background:#121722; }
    </style>
    """,
    unsafe_allow_html=True,
)

lang = st.sidebar.selectbox("Language / Язык / Til", ["ru", "uz", "en"], index=0)
t = TEXTS[lang]

st.markdown(f"<p class='title'>{t['title']}</p>", unsafe_allow_html=True)
st.markdown(f"<p class='subtitle'>{t['subtitle']}</p>", unsafe_allow_html=True)
st.markdown(f"<div style='text-align:center; margin-bottom:10px;'><span class='badge'>{t['badge']}</span></div>", unsafe_allow_html=True)

st.info(t["disclaimer"])

with st.expander(t["guide_title"], expanded=True):
    st.markdown(t["guide_md"])

c_ru, c_uz = st.columns(2)
with c_ru:
    st.markdown(f"### {TEXTS['ru']['ru_help_title']}")
    st.markdown(TEXTS["ru"]["ru_help_md"])
with c_uz:
    st.markdown(f"### {TEXTS['uz']['uz_help_title']}")
    st.markdown(TEXTS["uz"]["uz_help_md"])


@st.cache_data
def compute_reference_hist(data_dir: str = "data/classification/train", max_samples: int = 700):
    root = Path(data_dir)
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
        acc = hist if acc is None else acc + hist

    if acc is None:
        return None
    acc /= (np.sum(acc) + 1e-7)
    return acc


def is_non_biopsy(image: Image.Image, hist_corr_threshold: float = 0.25):
    arr = np.array(image.convert("RGB"))
    h, w = arr.shape[:2]
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Strong cue: frontal face detected -> out-of-domain for biopsy task.
    face_model = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
    if len(faces) > 0:
        return True, "Face-like content detected (out-of-domain)."

    ref_hist = compute_reference_hist()
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


@st.cache_resource
def load_cls_model(model_path: str):
    ckpt = torch.load(model_path, map_location="cpu")
    model = timm.create_model(ckpt.get("model_name", "efficientnetv2_rw_s"), pretrained=False, num_classes=12)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    img_size = ckpt.get("img_size", 224)
    return model, img_size


@st.cache_resource
def load_seg_model(model_path: str):
    ckpt = torch.load(model_path, map_location="cpu")
    model = smp.Unet(
        encoder_name=ckpt.get("encoder_name", "efficientnet-b3"),
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    img_size = ckpt.get("img_size", 256)
    return model, img_size


def predict_class(image: Image.Image, model, img_size: int):
    tfm = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    x = tfm(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].numpy()
    pred = int(np.argmax(probs))
    conf = float(probs[pred])
    return pred, conf


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


def predict_mask(
    image: Image.Image,
    model,
    img_size: int,
    threshold_mode: str = "fixed",
    threshold: float = 0.5,
    percentile: float = 85.0,
    min_area: int = 80,
    max_components: int = 3,
):
    arr = np.array(image.convert("RGB"))
    h, w = arr.shape[:2]
    x = cv2.resize(arr, (img_size, img_size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    x = torch.tensor(x).unsqueeze(0)
    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0, 0].numpy()
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
        mask2 = postprocess_mask(mask2, min_area=max(20, min_area // 3), max_components=max_components)
        coverage2 = float((mask2 > 0).mean())
        if coverage2 > coverage:
            mask = mask2
            coverage = coverage2

    return mask, prob_resized, coverage, thr


left, right = st.columns([1, 1], gap="large")
with left:
    st.markdown("<div class='block'>", unsafe_allow_html=True)
    uploaded = st.file_uploader(t["upload"], type=["png", "jpg", "jpeg"])
    cls_ckpt = st.text_input(t["cls_model"], "models/classification/classifier.pt")
    seg_ckpt = st.text_input(t["seg_model"], "models/segmentation/segmenter.pt")
    threshold_mode = st.selectbox(t["thr_mode"], ["fixed", "otsu", "percentile"], index=0)
    threshold = st.slider(t["thr"], min_value=0.10, max_value=0.95, value=0.65, step=0.05)
    percentile = st.slider(t["perc"], min_value=50, max_value=99, value=85, step=1)
    min_area = st.slider(t["min_area"], min_value=5, max_value=500, value=40, step=5)
    max_components = st.slider(t["max_comp"], min_value=1, max_value=10, value=3, step=1)
    run = st.button(t["run"], type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='block'>", unsafe_allow_html=True)
    st.write(t["results"])
    if uploaded is None:
        st.write(t["upload_hint"])
    st.markdown("</div>", unsafe_allow_html=True)

if run and uploaded is not None:
    image = Image.open(uploaded)
    non_biopsy, reason = is_non_biopsy(image)
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Input", use_container_width=True)
    with col2:
        if non_biopsy:
            st.metric(t["pred_class"], t["non_biopsy"], t["rejected"])
            st.warning(f"{t['quality']}: {reason}")
        elif Path(cls_ckpt).exists():
            cls_model, cls_size = load_cls_model(cls_ckpt)
            pred, conf = predict_class(image, cls_model, cls_size)
            st.metric(t["pred_class"], str(pred), f"{conf*100:.2f}% {t['confidence']}")
            st.caption(reason)
        else:
            st.warning(t["model_nf_cls"])

    if non_biopsy:
        st.warning(t["skip_seg"])
    elif Path(seg_ckpt).exists():
        seg_model, seg_size = load_seg_model(seg_ckpt)
        mask, prob_map, coverage, used_thr = predict_mask(
            image,
            seg_model,
            seg_size,
            threshold_mode=threshold_mode,
            threshold=threshold,
            percentile=float(percentile),
            min_area=min_area,
            max_components=max_components,
        )
        overlay = np.array(image.convert("RGB")).copy()
        overlay[mask > 0] = [255, 80, 80]
        blend = cv2.addWeighted(np.array(image.convert("RGB")), 0.65, overlay, 0.35, 0)
        contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_overlay = np.array(image.convert("RGB")).copy()
        cv2.drawContours(contour_overlay, contours, -1, (50, 255, 120), 2)
        dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
        if np.max(dist) > 0:
            dist = (dist / np.max(dist) * 255.0).astype(np.uint8)
        else:
            dist = np.zeros_like(mask, dtype=np.uint8)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(mask, caption=t["mask"], use_container_width=True, clamp=True)
        with c2:
            st.image(blend, caption=t["overlay"], use_container_width=True)
        with c3:
            st.image((prob_map * 255).astype(np.uint8), caption=t["prob"], use_container_width=True, clamp=True)
            st.image(contour_overlay, caption=t["contour"], use_container_width=True)
        st.image(dist, caption=t["dist"], use_container_width=True, clamp=True)
        st.caption(f"{t['coverage']}: {coverage*100:.2f}% | {t['thr_used']}: {used_thr:.3f} ({threshold_mode})")
    else:
        st.warning(t["model_nf_seg"])
