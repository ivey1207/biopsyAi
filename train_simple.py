import torch
import torch.nn as nn
import time

print("Script started!")
device = torch.device("cpu")
print(f"Using device: {device}")

model = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 1, 3, padding=1)
)
model.to(device)
print("Model created!")

x = torch.randn(1, 3, 224, 224).to(device)
y = model(x)
print(f"Output shape: {y.shape}")

print("Success!")
