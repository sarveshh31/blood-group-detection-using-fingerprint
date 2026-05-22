"""
feature_extractor.py
--------------------
EfficientNet-B0 based CNN feature extractor.
Uses pretrained ImageNet weights — no fine-tuning.
Supports GPU acceleration and batch processing.
"""

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from tqdm import tqdm
import os


# ─── ImageNet normalization constants ────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ─── Device setup ─────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    return device


# ─── CNN Feature Extractor ────────────────────────────────────────────────────
def build_feature_extractor(device: torch.device) -> nn.Module:
    """
    Load EfficientNet-B0 with pretrained ImageNet weights.
    Replace the classifier head with Identity to extract 1280-dim features.
    """
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)

    # Replace classifier → raw feature vector (1280-dim)
    model.classifier = nn.Identity()

    model = model.to(device)
    model.eval()  # Freeze BN layers too
    print("[INFO] EfficientNet-B0 loaded (classifier removed, feature extraction mode)")
    return model


# ─── Dataset wrapper ──────────────────────────────────────────────────────────
class FingerprintDataset(Dataset):
    """Wraps a list of (image_path, label_index) pairs."""

    def __init__(self, samples: list, transform=None):
        self.samples   = samples          # [(path, label_idx), ...]
        self.transform = transform or TRANSFORM

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Could not open {path}: {e}. Using blank image.")
            img = Image.new("RGB", (224, 224), color=0)
        return self.transform(img), label


# ─── Batch feature extraction ─────────────────────────────────────────────────
def extract_features(
    model: nn.Module,
    samples: list,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract CNN features for all samples using batch DataLoader.

    Returns
    -------
    features : np.ndarray  shape (N, 1280)
    labels   : np.ndarray  shape (N,)
    """
    dataset = FingerprintDataset(samples)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    all_features, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Extracting features", unit="batch"):
            images = images.to(device)
            feats  = model(images)                  # (B, 1280)
            all_features.append(feats.cpu().numpy())
            all_labels.append(labels.numpy())

    features = np.concatenate(all_features, axis=0)
    labels   = np.concatenate(all_labels,   axis=0)
    print(f"[INFO] Feature matrix: {features.shape}")
    return features, labels


# ─── Single-image inference ───────────────────────────────────────────────────
def extract_single_image(
    model: nn.Module,
    image_path: str,
    device: torch.device,
) -> np.ndarray:
    """
    Extract features from one image for inference.

    Returns
    -------
    feature : np.ndarray  shape (1, 1280)
    """
    img = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0).to(device)   # (1, 3, 224, 224)
    with torch.no_grad():
        feat = model(tensor).cpu().numpy()             # (1, 1280)
    return feat


def extract_single_pil(
    model: nn.Module,
    pil_image: Image.Image,
    device: torch.device,
) -> np.ndarray:
    """Extract features from a PIL Image object (used by Flask)."""
    img = pil_image.convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(tensor).cpu().numpy()
    return feat
