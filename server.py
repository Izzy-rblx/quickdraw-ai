import os
import io
import base64
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
import tensorflow as tf
import requests

app = Flask(__name__)

# GitHub Release model URL
MODEL_URL = "https://github.com/Izzy-rblx/quickdraw-ai/releases/download/v1/quickdraw_model.keras"
MODEL_PATH = "quickdraw_model.keras"

# Download model if missing
if not os.path.exists(MODEL_PATH):
    print(f"⬇️ Downloading model from {MODEL_URL}...")
    r = requests.get(MODEL_URL, stream=True)
    if r.status_code == 200:
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("✅ Model downloaded successfully")
    else:
        raise RuntimeError(f"❌ Failed to download model: HTTP {r.status_code}")

# Load the model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ ERROR: Failed to load model from '{MODEL_PATH}': {e}")
    raise

# Load categories
CATEGORIES_FILE = "categories.txt"
if os.path.exists(CATEGORIES_FILE):
    with open(CATEGORIES_FILE, "r") as f:
        categories = [line.strip() for line in f.readlines()]
else:
    categories = []
    print("⚠️ WARNING: categories.txt not found. Predictions will not have labels.")

def preprocess_image(image_data):
    """Preprocess input image (base64 or file) to model format"""
    image = Image.open(io.BytesIO(image_data)).convert("L")  # grayscale
    image = image.resize((28, 28))  # resize to model input
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)
    return image_array

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" in request.files:
            image_data = request.files["file"].read()
        else:
            data = request.get_json()
            if "image" not in data:
                return jsonify({"error": "No image provided"}), 400
            image_data = base64.b64decode(data["image"])

        processed = preprocess_image(image_data)
        predictions = model.predict(processed)[0]
        top_index = int(np.argmax(predictions))
        confidence = float(np.max(predictions))
        label = categories[top_index] if categories else str(top_index)

        return jsonify({
            "label": label,
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "✅ QuickDraw AI API is running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
