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

def sha256sum(filename):
    """Calculate SHA256 of a file."""
    h = hashlib.sha256()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def verify_model():
    """Check if model file matches expected checksum."""
    if not os.path.exists(MODEL_PATH):
        return False
    checksum = sha256sum(MODEL_PATH)
    if checksum.lower() == MODEL_SHA256.lower():
        print(f"✅ Model checksum verified: {checksum}")
        return True
    else:
        print(f"❌ Model checksum mismatch! Expected {MODEL_SHA256}, got {checksum}")
        return False

def download_model():
    """Download model from GitHub Releases."""
    print(f"⬇️ Downloading model from {GITHUB_RELEASE_URL} ...")
    response = requests.get(GITHUB_RELEASE_URL, stream=True)
    if response.status_code == 200:
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(1024 * 1024):
                f.write(chunk)
        print(f"✅ Model downloaded to {MODEL_PATH}")
    else:
        print(f"❌ Failed to download model: HTTP {response.status_code}")
        sys.exit(1)

def load_model_safe():
    """Try loading model, download + verify if missing/broken."""
    abs_path = os.path.abspath(MODEL_PATH)
    print(f"🔍 Looking for model at: {abs_path}")
    print(f"📂 Current working directory: {os.getcwd()}")
    print(f"📄 Files in CWD: {os.listdir(os.getcwd())}")

    if not verify_model():
        print("⚠️ Local model missing or invalid. Downloading fresh copy...")
        download_model()
        if not verify_model():
            print("❌ Downloaded model is corrupted. Exiting.")
            sys.exit(1)

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully.")
        return model
    except Exception as e:
        print(f"❌ ERROR loading model: {e}")
        traceback.print_exc()
        sys.exit(1)

model = load_model_safe()

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        img = Image.open(request.files["image"]).convert("L").resize((28, 28))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        preds = model.predict(img_array)
        pred_class = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0]))

        return jsonify({"class": pred_class, "confidence": confidence})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "QuickDraw AI is running!", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
