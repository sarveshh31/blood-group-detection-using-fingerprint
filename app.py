"""
app.py
------
Flask backend for blood group prediction from fingerprint images.

Run:
    python app.py
Then open:  http://127.0.0.1:5000
"""

import os
import sys
import base64
import io
from flask import Flask, request, jsonify, render_template
from PIL import Image

# Make src/ importable from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from predict import predict_from_pil, load_pipeline

# ─── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB upload limit

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "tiff", "webp"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ─── Pre-load models at startup ───────────────────────────────────────────────
print("[APP] Loading inference pipeline …")
try:
    load_pipeline()
    print("[APP] Models ready ✓")
except FileNotFoundError as e:
    print(f"[APP] WARNING: {e}")
    print("[APP] Start the server anyway; prediction will fail until models are trained.")


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # ── Validate upload ──────────────────────────────────────────────────────
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400

    # ── Read image ───────────────────────────────────────────────────────────
    try:
        img_bytes = file.read()
        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Cannot read image: {str(e)}"}), 400

    # ── Predict ──────────────────────────────────────────────────────────────
    try:
        result = predict_from_pil(pil_image)
    except FileNotFoundError as e:
        return jsonify({
            "error": "Models not found. Please run  python src/train.py  first."
        }), 503
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

    # ── Build response ───────────────────────────────────────────────────────
    # Encode image as base64 for preview in UI
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return jsonify({
        "predicted_class":  result["predicted_class"],
        "confidence":       round(result["confidence"] * 100, 1) if result["confidence"] else None,
        "all_probabilities": {
            cls: round(p * 100, 1)
            for cls, p in (result["all_probabilities"] or {}).items()
        },
        "image_preview": f"data:image/jpeg;base64,{img_b64}",
    })


# ─── Health check ─────────────────────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
