#!/usr/bin/env python3
"""
Evaluation script for MAX78000 Handwash Detection Model
-------------------------------------------------------
This version evaluates a pretrained model on the test dataset
and computes classification metrics using scikit-learn.

Usage:
    python train_sklearn.py \
        --model ai85handwashnet64 \
        --dataset handwash64 \
        --data data/handwash64 \
        --exp-load-weights-from ../ai8x-synthesis/trained/ai85-handwash-qat8-q.pth.tar \
        --evaluate \
        --confusion
"""

import argparse
import os

# Make sure Python can find the models folder
sys.path.append(os.path.join(os.path.dirname(__file__), "models"))
try:
    from ai85net-handwash64 import ai85net_handwash64
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "❌ Could not import 'ai85net-handwash64'. "
        "Make sure models/ai85net-handwash64.py exists and has ai85net_handwash64()."
    )

import torch
import torch.nn as nn   
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# -------------------------------------------------------------------------
# Example placeholder for ai85handwashnet64
# Replace this import with your actual model if available
# from models.ai85handwashnet64 import ai85handwashnet64
# -------------------------------------------------------------------------
# class DummyHandwashNet(nn.Module):
#     def __init__(self, num_classes=4):
#         super().__init__()
#         self.flatten = nn.Flatten()
#         self.fc = nn.Linear(64 * 64 * 3, num_classes)

#     def forward(self, x):
#         x = self.flatten(x)
#         return self.fc(x)

# -------------------------------------------------------------------------
# Argument Parser
# -------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Evaluate model using sklearn metrics")

parser.add_argument("--model", type=str, default="ai85handwashnet64",
                    help="Model architecture name (for display)")
parser.add_argument("--dataset", type=str, default="handwash64",
                    help="Dataset name (for display)")
parser.add_argument("--data", type=str, required=True,
                    help="Path to dataset directory")
parser.add_argument("--exp-load-weights-from", type=str, required=True,
                    help="Path to pretrained weights (.pth or .pth.tar)")
parser.add_argument("--evaluate", action="store_true",
                    help="Run in evaluation mode")
parser.add_argument("--confusion", action="store_true",
                    help="Show confusion matrix and classification report")
parser.add_argument("--device", type=str, default="MAX78000",
                    help="Device type (for display only)")
parser.add_argument("--batch-size", type=int, default=32,
                    help="Batch size for evaluation")

args = parser.parse_args()

# -------------------------------------------------------------------------
# Device configuration
# -------------------------------------------------------------------------
device = torch.device("cpu")
print(f"\n🧠 Evaluating on device: {args.device} (using CPU)\n")

# -------------------------------------------------------------------------
# Load dataset
# -------------------------------------------------------------------------
data_dir = args.data
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Dataset path not found: {data_dir}")

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
class_names = test_dataset.classes
num_classes = len(class_names)

print(f"📂 Loaded test dataset: {len(test_dataset)} images, {num_classes} classes\n")

# -------------------------------------------------------------------------
# Load model
# -------------------------------------------------------------------------
# Replace DummyHandwashNet with your actual model import if available
model = ai85handwashnet64(pretrained=False, num_classes=num_classes)
model.to(device)

if not os.path.exists(args.exp_load_weights_from):
    raise FileNotFoundError(f"Checkpoint not found: {args.exp_load_weights_from}")

print(f"🔹 Loading pretrained weights from: {args.exp_load_weights_from}")
checkpoint = torch.load(args.exp_load_weights_from, map_location=device)

# Handle common checkpoint formats
if "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"], strict=False)
else:
    model.load_state_dict(checkpoint, strict=False)

model.eval()

# -------------------------------------------------------------------------
# Evaluate model
# -------------------------------------------------------------------------
print("🔎 Running evaluation on test dataset...\n")
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# -------------------------------------------------------------------------
# Compute metrics using sklearn
# -------------------------------------------------------------------------
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

print("📊 Evaluation Results:")
print(f"  • Accuracy : {accuracy * 100:.2f}%")
print(f"  • Precision: {precision * 100:.2f}%")
print(f"  • Recall   : {recall * 100:.2f}%")
print(f"  • F1-Score : {f1 * 100:.2f}%")

# -------------------------------------------------------------------------
# Optional: Confusion matrix and detailed report
# -------------------------------------------------------------------------
if args.confusion:
    cm = confusion_matrix(y_true, y_pred)
    print("\n🧾 Confusion Matrix:")
    print(cm)
    print("\n📄 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

print("\n✅ Evaluation complete.\n")
