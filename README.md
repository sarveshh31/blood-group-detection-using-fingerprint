# 🩸 Blood Group Detection Using Fingerprint Images

An AI-powered blood group detection system that predicts a person's blood group from fingerprint images using Machine Learning and Deep Learning techniques. The project uses feature extraction, preprocessing, and trained classification models to identify blood groups accurately from biometric patterns.

---

# 🚀 Features

- 🔍 Blood group prediction using fingerprint images
- 🧠 Machine Learning + Deep Learning models
- 📊 Trained SVM and PyTorch models included
- ⚡ Fast prediction pipeline
- 🌐 Flask web application support
- 📁 Organized dataset structure
- 🧪 Testing dataset for evaluation
- 💾 Saved PCA, scaler, and class label encoders

---

# 🛠️ Tech Stack

- Python
- PyTorch
- Scikit-learn
- OpenCV
- NumPy
- Flask
- HTML/CSS

---

# 📂 Project Structure

```bash
BLOOD-GROUP-DETECTION-USING-FINGERPRINT/
│
├── dataset_blood_group/
│   ├── A+
│   ├── A-
│   ├── AB+
│   ├── AB-
│   ├── B+
│   ├── B-
│   ├── O+
│   └── O-
│
├── models/
│   ├── best_model.pth
│   ├── class_names.pkl
│   ├── pca.pkl
│   ├── scaler.pkl
│   └── svm_model.pkl
│
├── src/
│   ├── dataset.py
│   ├── feature_extractor.py
│   ├── model.py
│   ├── predict.py
│   ├── train.py
│   └── utils.py
│
├── templates/
│
├── testing_dataset/
│
├── venv/
│
├── .gitignore
├── app.py
├── README.md
└── requirements.txt
```

---

# ⚙️ Installation

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/Blood-Group-Detection-Using-Fingerprint.git
cd Blood-Group-Detection-Using-Fingerprint
```

---

## 2️⃣ Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Application

Start the Flask application:

```bash
python app.py
```

Open your browser and go to:

```bash
http://127.0.0.1:5000
```

---

# 🧠 Model Training

To train the model again:

```bash
python src/train.py
```

---

# 🔮 Prediction

To predict blood group using fingerprint images:

```bash
python src/predict.py
```

---

# 📊 Dataset Information

The dataset contains fingerprint images categorized into the following blood groups:

- A+
- A-
- B+
- B-
- AB+
- AB-
- O+
- O-

Each folder inside `dataset_blood_group` represents a separate blood group class.

---

# 🧪 Machine Learning Pipeline

The project follows the following workflow:

1. Fingerprint image preprocessing
2. Feature extraction
3. Feature scaling using StandardScaler
4. PCA dimensionality reduction
5. Classification using:
   - Support Vector Machine (SVM)
   - Deep Learning Model (PyTorch)

---

# 🌐 Web Application

The project includes a Flask-based web application where users can:

- Upload fingerprint images
- Predict blood groups
- View prediction results instantly

---

# 📸 Future Improvements

- Improve model accuracy with larger datasets
- Add CNN-based architecture
- Deploy using Docker
- Add real-time fingerprint scanner integration
- Convert into Android/iOS mobile application
- Add cloud deployment support

---

# 🤝 Contributing

Contributions are welcome.

## Steps to Contribute

1. Fork the repository

2. Create a feature branch

```bash
git checkout -b feature-name
```

3. Commit your changes

```bash
git commit -m "Added new feature"
```

4. Push to GitHub

```bash
git push origin feature-name
```

5. Create a Pull Request

---

# 📜 License

This project is developed for educational and research purposes only.

---

# 👨‍💻 Author

## Sarvesh Tiwari & Chaitanya Wanjarkar.

---

# ⭐ Support

If you found this project useful, give it a ⭐ on GitHub.

---