import os
import sys
import hashlib
import traceback
import requests
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

MODEL_PATH = "quickdraw_model.keras"
GITHUB_RELEASE_URL = "https://github.com/Izzy-rblx/quickdraw-ai/releases/download/v1/quickdraw_model.keras"
MODEL_SHA256 = "af274f007abc6d93ee760177affbc37b3b1674cefd811389cae661017dcd6784"

def log(msg):
    """Helper to print logs with a prefix so they’re easy to spot in Render logs."""
    print(f"[SERVER LOG] {msg}", flush=True)

def sha256sum(filename):
    """Calculate SHA256 of a file."""
    h = hashlib.sha256()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def verify_model():
    """Check if model file exists and matches expected checksum."""
    if not os.path.exists(MODEL_PATH):
        log("❌ Model file not found.")
        return False

    size = os.path.getsize(MODEL_PATH)
    log(f"📦 Found model file at {MODEL_PATH}, size: {size / (1024*1024):.2f} MB")

    checksum = sha256sum(MODEL_PATH)
    if checksum.lower() == MODEL_SHA256.lower():
        log(f"✅ Model checksum verified: {checksum}")
        return True
    else:
        log(f"❌ Model checksum mismatch! Expected {MODEL_SHA256}, got {checksum}")
        return False

def download_model():
    """Download model from GitHub Releases."""
    log(f"⬇️ Downloading model from {GITHUB_RELEASE_URL} ...")
    response = requests.get(GITHUB_RELEASE_URL, stream=True)
    if response.status_code == 200:
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(1024 * 1024):
                f.write(chunk)
        log(f"✅ Model downloaded to {MODEL_PATH}")
    else:
        log(f"❌ Failed to download model: HTTP {response.status_code}")
        sys.exit(1)

def load_model_safe():
    """Try loading model, download + verify if missing/broken."""
    abs_path = os.path.abspath(MODEL_PATH)
    log(f"🔍 Looking for model at: {abs_path}")
    log(f"📂 Current working directory: {os.getcwd()}")
    log(f"📄 Files in CWD: {os.listdir(os.getcwd())}")

    if not verify_model():
        log("⚠️ Local model missing or invalid. Downloading fresh copy...")
        download_model()
        if not verify_model():
            log("❌ Downloaded model is corrupted. Exiting.")
            sys.exit(1)

    try:
        log("🔄 Attempting to load model into TensorFlow...")
        model = tf.keras.models.load_model(MODEL_PATH)
        log("✅ Model loaded successfully into memory.")
        return model
    except Exception as e:
        log(f"❌ ERROR loading model: {e}")
        traceback.print_exc()
        sys.exit(1)

# Load model on startup
model = load_model_safe()

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            log("❌ Prediction request missing 'image' field.")
            return jsonify({"error": "No image provided"}), 400

        log("📥 Received image for prediction.")
        img = Image.open(request.files["image"]).convert("L").resize((28, 28))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        log("🔮 Running prediction on model...")
        preds = model.predict(img_array)
        pred_class = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0]))

        log(f"✅ Prediction done. Class={pred_class}, Confidence={confidence:.4f}")
        return jsonify({"class": pred_class, "confidence": confidence})
    except Exception as e:
        log(f"❌ Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    log("📡 Health check hit: '/' endpoint called.")
    return "QuickDraw AI is running!", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    log(f"🚀 Starting Flask server on port {port} ...")
    app.run(host="0.0.0.0", port=port)
