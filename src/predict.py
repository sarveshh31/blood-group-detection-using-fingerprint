"""
predict.py
----------
Single-image prediction using trained SVM + EfficientNet feature extractor.

Usage (CLI):
    python src/predict.py path/to/fingerprint.jpg
"""

import os
import sys
import joblib
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_extractor import (
    get_device,
    build_feature_extractor,
    extract_single_image,
    extract_single_pil,
)

MODELS_DIR = "models"


# ─── Model loader (cached for Flask reuse) ────────────────────────────────────
_cache: dict = {}


def load_pipeline() -> tuple:
    """
    Load and cache the inference pipeline components:
      - CNN feature extractor
      - StandardScaler
      - SVM classifier
      - class names list
    """
    if _cache:
        return (
            _cache["model"],
            _cache["scaler"],
            _cache["svm"],
            _cache["class_names"],
            _cache["device"],
        )

    device = get_device()
    model  = build_feature_extractor(device)

    svm_path    = os.path.join(MODELS_DIR, "svm_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    meta_path   = os.path.join(MODELS_DIR, "class_names.pkl")

    for p in (svm_path, scaler_path, meta_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Model file not found: {p}\n"
                "Please run  python src/train.py  first."
            )

    svm         = joblib.load(svm_path)
    scaler      = joblib.load(scaler_path)
    class_names = joblib.load(meta_path)

    _cache.update(dict(
        model=model, scaler=scaler, svm=svm,
        class_names=class_names, device=device,
    ))
    print("[INFO] Inference pipeline loaded successfully.")
    return model, scaler, svm, class_names, device


# ─── Prediction helpers ───────────────────────────────────────────────────────
def predict_from_path(image_path: str) -> dict:
    """
    Predict blood group from a file path.

    Returns dict with keys: predicted_class, confidence, all_probabilities
    """
    model, scaler, svm, class_names, device = load_pipeline()

    feat = extract_single_image(model, image_path, device)   # (1, 1280)
    feat = scaler.transform(feat)

    pred_idx    = svm.predict(feat)[0]
    predicted   = class_names[pred_idx]

    probas = None
    confidence = None
    if hasattr(svm, "predict_proba"):
        probas     = svm.predict_proba(feat)[0]
        confidence = float(probas[pred_idx])

    return {
        "predicted_class":  predicted,
        "confidence":       confidence,
        "all_probabilities": {
            class_names[i]: float(p)
            for i, p in enumerate(probas)
        } if probas is not None else None,
    }


def predict_from_pil(pil_image: Image.Image) -> dict:
    """
    Predict blood group from a PIL Image object.
    Used by the Flask backend to avoid writing temp files.
    """
    model, scaler, svm, class_names, device = load_pipeline()

    feat = extract_single_pil(model, pil_image, device)      # (1, 1280)
    feat = scaler.transform(feat)

    pred_idx  = svm.predict(feat)[0]
    predicted = class_names[pred_idx]

    probas = None
    confidence = None
    if hasattr(svm, "predict_proba"):
        probas     = svm.predict_proba(feat)[0]
        confidence = float(probas[pred_idx])

    return {
        "predicted_class":  predicted,
        "confidence":       confidence,
        "all_probabilities": {
            class_names[i]: float(p)
            for i, p in enumerate(probas)
        } if probas is not None else None,
    }


# ─── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    if not os.path.isfile(img_path):
        print(f"[ERROR] File not found: {img_path}")
        sys.exit(1)

    result = predict_from_path(img_path)

    print("\n" + "=" * 40)
    print(f"  Predicted Blood Group : {result['predicted_class']}")
    if result["confidence"] is not None:
        print(f"  Confidence            : {result['confidence'] * 100:.1f}%")
    print("=" * 40)

    if result["all_probabilities"]:
        print("\nProbabilities per class:")
        for cls, prob in sorted(result["all_probabilities"].items(),
                                key=lambda x: -x[1]):
            bar = "█" * int(prob * 30)
            print(f"  {cls:>4s}  {bar:<30s}  {prob * 100:5.1f}%")
