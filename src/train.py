"""
train.py
--------
End-to-end training pipeline:
  1. Load dataset from  dataset_blood_group/<class>/...
  2. Extract CNN features (EfficientNet-B0, no grad)
  3. StandardScaler → SVM (RBF kernel)
  4. Evaluate & print metrics
  5. Save models/svm_model.pkl  and  models/scaler.pkl
"""

import os
import sys
import glob
import joblib
import numpy as np

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Make sure src/ is on the path when running as  python src/train.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_extractor import (
    get_device,
    build_feature_extractor,
    extract_features,
)

# ─── Configuration ─────────────────────────────────────────────────────────────
DATASET_DIR  = "dataset_blood_group"
MODELS_DIR   = "models"
TEST_SIZE    = 0.20
RANDOM_STATE = 42
BATCH_SIZE   = 64          # increase if you have more RAM / GPU VRAM
NUM_WORKERS  = 0           # set >0 on Linux for faster loading

# Blood group class names (order must match folder names)
CLASSES = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]

# SVM hyper-parameters
SVM_PARAMS = dict(
    kernel       = "rbf",
    C            = 5,
    gamma        = "scale",
    class_weight = "balanced",
    probability  = True,       # enables predict_proba — useful for confidence scores
    random_state = RANDOM_STATE,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────
def load_dataset(dataset_dir: str) -> tuple[list, list[str]]:
    """
    Walk dataset_dir and collect (image_path, label_index) pairs.
    Folder names are used as class labels.
    """
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(
            f"Dataset directory '{dataset_dir}' not found. "
            "Run this script from the project root."
        )

    class_dirs = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])

    if not class_dirs:
        raise ValueError(f"No sub-folders found inside '{dataset_dir}'.")

    print(f"[INFO] Found {len(class_dirs)} classes: {class_dirs}")

    samples   = []
    class_map = {name: idx for idx, name in enumerate(class_dirs)}

    for class_name in class_dirs:
        class_path = os.path.join(dataset_dir, class_name)
        images = (
            glob.glob(os.path.join(class_path, "*.jpg"))
            + glob.glob(os.path.join(class_path, "*.jpeg"))
            + glob.glob(os.path.join(class_path, "*.png"))
            + glob.glob(os.path.join(class_path, "*.bmp"))
        )
        label_idx = class_map[class_name]
        for img_path in images:
            samples.append((img_path, label_idx))
        print(f"  {class_name:>6s}  →  {len(images):>5d} images")

    print(f"[INFO] Total samples: {len(samples)}")
    return samples, class_dirs


def print_confusion_matrix(cm: np.ndarray, class_names: list[str]) -> None:
    """Pretty-print confusion matrix with class labels."""
    col_w = max(len(c) for c in class_names) + 2
    header = " " * col_w + "  ".join(f"{c:>{col_w}}" for c in class_names)
    print("\nConfusion Matrix (rows=actual, cols=predicted):")
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>{col_w}}" for v in row)
        print(f"{class_names[i]:>{col_w}}  {row_str}")
    print()


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Load dataset paths
    samples, class_names = load_dataset(DATASET_DIR)

    # 2. Stratified train/test split (on paths, before feature extraction)
    labels_all = [s[1] for s in samples]
    train_samples, test_samples = train_test_split(
        samples,
        test_size    = TEST_SIZE,
        stratify     = labels_all,
        random_state = RANDOM_STATE,
    )
    print(f"\n[INFO] Train: {len(train_samples)} | Test: {len(test_samples)}")

    # 3. Build CNN feature extractor
    device = get_device()
    model  = build_feature_extractor(device)

    # 4. Extract features
    print("\n[INFO] Extracting TRAIN features …")
    X_train, y_train = extract_features(model, train_samples, device, BATCH_SIZE, NUM_WORKERS)

    print("\n[INFO] Extracting TEST features …")
    X_test, y_test   = extract_features(model, test_samples,  device, BATCH_SIZE, NUM_WORKERS)

    # 5. Feature scaling
    print("\n[INFO] Fitting StandardScaler …")
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # 6. Train SVM
    print(f"\n[INFO] Training SVM  {SVM_PARAMS} …")
    svm = SVC(**SVM_PARAMS)
    svm.fit(X_train, y_train)
    print("[INFO] SVM training complete.")

    # 7. Evaluate
    y_pred = svm.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print(f"  Accuracy: {acc * 100:.2f}%")
    print("=" * 60)
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names = class_names,
        digits       = 4,
    ))

    cm = confusion_matrix(y_test, y_pred)
    print_confusion_matrix(cm, class_names)

    # 8. Save models
    svm_path    = os.path.join(MODELS_DIR, "svm_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    meta_path   = os.path.join(MODELS_DIR, "class_names.pkl")

    joblib.dump(svm,         svm_path)
    joblib.dump(scaler,      scaler_path)
    joblib.dump(class_names, meta_path)

    print(f"[INFO] Saved SVM    →  {svm_path}")
    print(f"[INFO] Saved scaler →  {scaler_path}")
    print(f"[INFO] Saved labels →  {meta_path}")
    print("\n✅ Training complete. Run  python app.py  to start the web UI.")


if __name__ == "__main__":
    main()
