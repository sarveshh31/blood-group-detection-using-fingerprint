# BloodPrint — Blood Group Classifier via Fingerprint
**CNN (EfficientNet-B0) + SVM hybrid pipeline with Flask web UI**

---

## Project Structure

```
blood_group_classifier/
├── dataset_blood_group/     ← your dataset goes here
│   ├── A+/
│   ├── A-/
│   ├── B+/
│   ├── B-/
│   ├── AB+/
│   ├── AB-/
│   ├── O+/
│   └── O-/
├── models/                  ← auto-created after training
│   ├── svm_model.pkl
│   ├── scaler.pkl
│   └── class_names.pkl
├── src/
│   ├── feature_extractor.py
│   ├── train.py
│   └── predict.py
├── templates/
│   └── index.html
├── app.py
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place your dataset
Make sure `dataset_blood_group/` exists in the project root with one sub-folder per blood group.

### 3. Train the model
```bash
python src/train.py
```
This will:
- Extract 1280-dim EfficientNet-B0 features for every image (GPU if available)
- Fit a StandardScaler
- Train an RBF-SVM (C=5, gamma='scale', class_weight='balanced')
- Print accuracy, classification report, confusion matrix
- Save `models/svm_model.pkl`, `models/scaler.pkl`, `models/class_names.pkl`

### 4. Launch the web UI
```bash
python app.py
```
Open your browser at **http://127.0.0.1:5000**

### 5. Predict from command line (optional)
```bash
python src/predict.py path/to/fingerprint.jpg
```

---

## Model Design

| Component | Details |
|-----------|---------|
| Feature extractor | EfficientNet-B0 (ImageNet pretrained, frozen) |
| Feature dimension | 1280 |
| Scaler | StandardScaler |
| Classifier | SVC(kernel='rbf', C=5, gamma='scale', class_weight='balanced') |
| Input size | 224 × 224 RGB |
| Split | 80% train / 20% test (stratified) |

---

## Expected Accuracy

~70–75% on 6 000 fingerprint images (8 classes).

---

## Notes

- No PCA, no GridSearchCV, no DL classifier — pure CNN features → SVM.
- `probability=True` in SVC enables per-class confidence scores in the UI.
- The Flask server pre-loads all models at startup for fast inference.
