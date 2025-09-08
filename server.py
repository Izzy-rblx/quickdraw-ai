import os
import requests
import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image

app = Flask(__name__)

# Model file settings
MODEL_PATH = "quickdraw_model.keras"
MODEL_URL = "https://github.com/Izzy-rblx/quickdraw-ai/releases/download/v1/quickdraw_model.keras"

# Download model if not already present
if not os.path.exists(MODEL_PATH):
    print(f"⚠️ {MODEL_PATH} not found, downloading from {MODEL_URL}...")
    r = requests.get(MODEL_URL)
    if r.status_code == 200:
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        print("✅ Model downloaded successfully.")
    else:
        raise RuntimeError(f"Failed to download model from {MODEL_URL}, status={r.status_code}")

# Load the model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ ERROR: Failed to load model: {e}")
    raise

# Load categories
CATEGORIES_PATH = "categories.txt"
if os.path.exists(CATEGORIES_PATH):
    with open(CATEGORIES_PATH, "r") as f:
        categories = [line.strip() for line in f.readlines()]
else:
    categories = [str(i) for i in range(10)]  # fallback

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("L").resize((28, 28))
    arr = np.array(image) / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    return arr

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        file = request.files["file"]
        image = Image.open(file.stream)
        input_arr = preprocess_image(image)

        preds = model.predict(input_arr)[0]
        idx = int(np.argmax(preds))
        confidence = float(preds[idx])

        return jsonify({
            "category": categories[idx] if idx < len(categories) else str(idx),
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def index():
    return jsonify({"status": "ok", "message": "Quickdraw AI API running"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
