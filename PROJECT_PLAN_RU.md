# AI in Healthcare Hackathon 2026 - Detailed Project Plan

## 1) What the technical assignment requires

Two mandatory tasks:
- Classification: train on `classification/train` and predict labels for all images in `classification/test` (1276 images, classes `0..11`).
- Segmentation: train binary mask model on `Segmentation/training` (+ validate on `Segmentation/validation`) and predict masks for `Segmentation/testing/images` (200 images).

Mandatory deliverables:
- `<TeamName> test_ground_truth.xlsx` with exactly 1276 rows (`Image_ID`, `Label`).
- Folder `<TeamName>/` with 200 predicted segmentation masks (`.png`, binary, same filename ID and image size).
- `models/` with saved models and inference-only scripts:
  - `models/classification/classify.py`
  - `models/segmentation/segment.py`
  - requirements files.

Competition constraints:
- No training on test sets.
- Pretrained models and transfer learning are allowed.
- Top-10 presentation requires working UI with upload + classification + segmentation visualization + disclaimer:
  - `For research and demonstration purposes only. Not for clinical use.`

Scoring:
- 70 points model score: classification accuracy (30) + segmentation mIoU (40).
- 30 points presentation quality (UI, robustness, clarity, creativity).

## 2) Dataset verification already completed

### Classification archive
- Test: 1276 files (matches requirement).
- Train class counts (imbalanced):
  - 0: 571
  - 1: 974
  - 2: 1043
  - 3: 750
  - 4: 814
  - 5: 441
  - 6: 545
  - 7: 2136
  - 8: 331
  - 9: 1111
  - 10: 899
  - 11: 1796
- Total train: 11411.
- Most images are ~`101x101` PNG (small images -> risk of overfitting/noise).

### Segmentation archive
- `Segmentation/training/images`: 1800
- `Segmentation/training/masks`: 1800
- `Segmentation/validation/images`: 400
- `Segmentation/validation/masks`: 400
- `Segmentation/testing/images`: 200
- Image/mask pairing is valid by stem (image `.jpg`, mask `.png`; same base ID like `937x`).
- Training image size sample: `128x128` (uniform, convenient for U-Net-like architectures).

## 3) Target architecture (fast and realistic for hackathon)

### A) Classification model
- Backbone: `timm` EfficientNetV2-S or ConvNeXt-Tiny pretrained on ImageNet.
- Input: upscale to `224x224`, normalize with ImageNet stats.
- Augmentations (Albumentations):
  - RandomResizedCrop, Horizontal/VerticalFlip, ColorJitter, GaussianNoise, CoarseDropout.
- Loss:
  - Weighted CrossEntropy or Focal Loss (because class imbalance is high).
- Training strategy:
  - Stratified K-Fold (5 folds) for robust generalization.
  - Mixed precision (`torch.cuda.amp`).
  - Cosine LR schedule + warmup.
  - Early stopping by macro-F1 / val accuracy.
- Inference:
  - TTA (flip-based).
  - Optional fold ensembling (best 3-5 checkpoints).

### B) Segmentation model
- Baseline: `segmentation_models_pytorch` U-Net with EfficientNet-B3 encoder.
- Alternative candidate: DeepLabV3+ (if time allows).
- Input pipeline:
  - Resize to `256x256` (or native `128x128` baseline).
  - Strong geometric + photometric augmentations.
- Loss:
  - Combo: `0.5 * BCEWithLogits + 0.5 * DiceLoss` (or focal+tversky).
- Metrics:
  - IoU and Dice on validation.
- Inference:
  - Sigmoid threshold tuning (0.35-0.65 search on validation).
  - Post-processing: remove tiny connected components, optional hole fill.
  - Output masks strictly binary PNG (`0` or `255`) with original size.

## 4) UI strategy (for Top-10 stage)

Stack:
- `Streamlit` (fastest for hackathon and demo reliability).

UI screens:
- Upload image.
- Run both models.
- Show:
  - predicted class ID + confidence.
  - segmentation overlay (image + mask heat/contour).
- Batch mode optional:
  - upload folder -> download zip/xlsx outputs.
- Required disclaimer visible on top/footer.

Reliability features:
- Input validation.
- Error handling with user-friendly messages.
- Inference latency display.

## 5) End-to-end folder blueprint (project workspace)

Recommended structure:
- `AI_Healthcare_Hackathon_2026/`
  - `data/`
    - `classification/`
    - `segmentation/`
  - `src/`
    - `classification/`
      - `train_cls.py`
      - `infer_cls.py`
      - `dataset.py`
      - `models.py`
      - `utils.py`
    - `segmentation/`
      - `train_seg.py`
      - `infer_seg.py`
      - `dataset.py`
      - `models.py`
      - `postprocess.py`
    - `common/`
      - `seed.py`
      - `io.py`
  - `app/`
    - `streamlit_app.py`
  - `submissions/`
    - `<TeamName> test_ground_truth.xlsx`
    - `<TeamName>/` (200 masks)
  - `models/`
    - `classification/`
    - `segmentation/`
  - `requirements.txt`
  - `Makefile`
  - `README.md`

## 6) Detailed execution plan (hour-by-hour style)

### Phase 0 - Setup (0.5-1 hour)
1. Create isolated environment.
2. Install dependencies:
   - `torch`, `torchvision`, `timm`, `albumentations`, `opencv-python`, `pandas`, `openpyxl`, `segmentation-models-pytorch`, `streamlit`, `scikit-learn`.
3. Unzip datasets into `data/`.
4. Run dataset integrity script and save report.

### Phase 1 - Classification baseline to strong model (2-3 hours)
1. Train quick baseline (single fold) for sanity.
2. Add class weights and stronger augmentations.
3. Run 5-fold CV training.
4. Export OOF metrics and confusion matrix.
5. Run test inference + TTA + ensemble.
6. Generate final Excel submission file.

### Phase 2 - Segmentation baseline to optimized model (2-3 hours)
1. Train baseline U-Net quickly.
2. Track IoU/Dice on validation.
3. Tune threshold and post-processing.
4. Train second architecture or second seed.
5. Ensemble best masks if beneficial.
6. Generate 200 binary PNG masks.

### Phase 3 - Inference-only scripts for organizers (1-1.5 hours)
1. `classify.py`:
   - Input test folder + saved model(s)
   - Output Excel exactly in requested format.
2. `segment.py`:
   - Input image folder + saved model(s)
   - Output folder of binary PNG masks with exact names and sizes.
3. Ensure no retraining inside scripts.

### Phase 4 - UI and demo hardening (1.5-2.5 hours)
1. Implement Streamlit app with dual inference.
2. Add visualization and confidence output.
3. Add robust error handling and loading states.
4. Add mandatory disclaimer text.
5. Prepare demo script in English (10-12 min).

### Phase 5 - Final QA and packaging (1 hour)
1. Validate submission structure exactly matches guideline.
2. Re-run inference scripts from clean session.
3. Confirm all rows/masks present and named correctly.
4. Package final folder and upload.

## 7) Critical risks and mitigations

Risk -> mitigation:
- Class imbalance in classification -> weighted loss + stratified folds + macro-F1 monitoring.
- Overfitting due to small images -> stronger augmentation, regularization, early stopping.
- Segmentation mask artifacts -> threshold tuning and morphology post-process.
- Deadline pressure -> lock baseline early, then improve iteratively.
- Demo failure -> keep lightweight local Streamlit app and fallback screenshots/video.

## 8) Immediate next actions (what to do right now)

1. Extract both zip files into:
   - `data/classification`
   - `data/segmentation`
2. Bootstrap training code for both tasks.
3. Run first classification and segmentation baselines in parallel.
4. Produce first valid submission artifacts quickly.
5. Iterate only on components that improve validation metrics.

---

If needed, this plan can be converted into a strict command-by-command checklist for execution in terminal (ready for immediate run).
