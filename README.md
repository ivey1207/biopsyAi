# AI Healthcare Hackathon 2026

This repository contains a practical baseline for:
- Image classification (`12` classes)
- Binary image segmentation
- Inference-only organizer scripts
- Demo UI (Streamlit)

## 1) Setup

```bash
cd "/Users/avazov/Desktop/AI_Healthcare_Hackathon_2026"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Dataset location

Expected paths:
- `data/classification/train`
- `data/classification/test`
- `data/Segmentation/training/images`
- `data/Segmentation/training/masks`
- `data/Segmentation/validation/images`
- `data/Segmentation/validation/masks`
- `data/Segmentation/testing/images`

## 3) Train models

Classification:
```bash
python src/classification/train_cls.py \
  --train-dir data/classification/train \
  --out models/classification/classifier.pt
```

Segmentation:
```bash
python src/segmentation/train_seg.py \
  --train-images data/Segmentation/training/images \
  --train-masks data/Segmentation/training/masks \
  --val-images data/Segmentation/validation/images \
  --val-masks data/Segmentation/validation/masks \
  --out models/segmentation/segmenter.pt
```

## 4) Organizer inference scripts

Classification xlsx:
```bash
python models/classification/classify.py \
  --images-dir data/classification/test \
  --model-path models/classification/classifier.pt \
  --output-xlsx submissions/TEAM_test_ground_truth.xlsx
```

Segmentation masks:
```bash
python models/segmentation/segment.py \
  --images-dir data/Segmentation/testing/images \
  --model-path models/segmentation/segmenter.pt \
  --output-dir submissions/TEAM \
  --threshold-mode fixed \
  --threshold 0.65 \
  --min-area 40 \
  --max-components 3
```

Adaptive examples:
```bash
# Otsu threshold from probability map
python models/segmentation/segment.py --images-dir data/Segmentation/testing/images --model-path models/segmentation/segmenter.pt --output-dir submissions/TEAM --threshold-mode otsu

# Percentile threshold (robust for hard images)
python models/segmentation/segment.py --images-dir data/Segmentation/testing/images --model-path models/segmentation/segmenter.pt --output-dir submissions/TEAM --threshold-mode percentile --percentile 85
```

## 5) Run UI demo

```bash
streamlit run app/streamlit_app.py
```

The UI includes the mandatory disclaimer:
`For research and demonstration purposes only. Not for clinical use.`
