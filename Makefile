PY=python3

.PHONY: verify train-cls train-seg infer-cls infer-seg ui api web-build web-preview

verify:
	$(PY) src/common/verify_data.py

train-cls:
	$(PY) src/classification/train_cls.py --train-dir data/classification/train --out models/classification/classifier.pt

train-seg:
	$(PY) src/segmentation/train_seg.py --train-images data/Segmentation/training/images --train-masks data/Segmentation/training/masks --val-images data/Segmentation/validation/images --val-masks data/Segmentation/validation/masks --out models/segmentation/segmenter.pt

infer-cls:
	$(PY) models/classification/classify.py --images-dir data/classification/test --model-path models/classification/classifier.pt --output-xlsx submissions/TEAM_test_ground_truth.xlsx

infer-seg:
	$(PY) models/segmentation/segment.py --images-dir data/Segmentation/testing/images --model-path models/segmentation/segmenter.pt --output-dir submissions/TEAM

api:
	./.venv/bin/uvicorn server.api_server:app --host 0.0.0.0 --port 8000

web-build:
	cd frontend && npm run build

web-preview:
	cd frontend && npm run preview
