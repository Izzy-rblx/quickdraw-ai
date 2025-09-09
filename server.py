import os
import sys
import traceback
import requests
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

MODEL_PATH = "quickdraw_model.keras"
GITHUB_RELEASE_URL = "https://github.com/Izzy-rblx/quickdraw-ai/releases/download/v1/quickdraw_model.keras"

def download_model():
    """Download model from GitHub Releases if missing or broken."""
    print(f"⬇️ Downloading model from {GITHUB_RELEASE_URL} ...")
    response = requests.get(GITHUB_RELEASE_URL, stream=True)
    if response.status_code == 200:
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(1024 * 1024):
                f.write(chunk)
        print(f"✅ Model downloaded and saved as {MODEL_PATH}")
    else:
        print(f"❌ Failed to download model: HTTP {response.status_code}")
        sys.exit(1)

def load_model_safe():
    """Try loading model, fallback to download if invalid."""
    abs_path = os.path.abspath(MODEL_PATH)
    print(f"🔍 Looking for model at: {abs_path}")
    print(f"📂 Current working directory: {os.getcwd()}")
    print(f"📄 Files in CWD: {os.listdir(os.getcwd())}")

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully from local file.")
        return model
    except Exception as e:
        print(f"❌ ERROR loading local model: {e}")
        print("Attempting to download model from GitHub Releases...")
        traceback.print_exc()
        download_model()
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully after download.")
        return model

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
