import os
import sys

print("Setting MPS_FORCE_DISABLE=1")
os.environ["MPS_FORCE_DISABLE"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

print(f"Python version: {sys.version}")

print("Importing torch...")
import torch
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")

print("Importing smp...")
import segmentation_models_pytorch as smp
print("SMP imported successfully!")

print("Importing cv2...")
import cv2
print(f"OpenCV: {cv2.__version__}")

print("All imports done!")
