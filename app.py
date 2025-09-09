import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)
CORS(app)

# ---------------------------
# Load categories
# ---------------------------
CATEGORIES_PATH = "categories.txt"
if os.path.exists(CATEGORIES_PATH):
    with open(CATEGORIES_PATH, "r") as f:
        categories = [line.strip() for line in f.readlines()]
    print(f"✅ Loaded {len(categories)} categories")
else:
    categories = []
    print("⚠️ No categories.txt found")

# ---------------------------
# Load model (SavedModel via TFSMLayer)
# ---------------------------
MODEL_PATH = "quickdraw_saved_model"
model = None
try:
    if os.path.exists(MODEL_PATH):
        model = keras.layers.TFSMLayer(MODEL_PATH, call_endpoint="serving_default")
        print("✅ Model loaded successfully (TFSMLayer)")
    else:
        print("⚠️ Model not found, predictions disabled")
except Exception as e:
    print("⚠️ Failed to load model:", e)


# ---------------------------
# Preprocess strokes → 28x28 grayscale
# ---------------------------
def preprocess_strokes(strokes, size=28):
    canvas = np.zeros((size, size), dtype=np.uint8)
    for stroke in strokes:
        for point in stroke:
            x = int(np.clip(point.get("x", 0) / 10, 0, size - 1))
            y = int(np.clip(point.get("y", 0) / 10, 0, size - 1))
            canvas[y, x] = 255
    img = canvas.astype("float32") / 255.0
    return img.reshape(1, size, size, 1)


# ---------------------------
# API Routes
# ---------------------------
@app.route("/")
def home():
    return "QuickDraw AI API is running (stroke-only mode)."


@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded"}), 503

    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No stroke data provided"}), 400

        strokes = data["image"]
        processed = preprocess_strokes(strokes)

        # Run inference
        preds = model(processed, training=False).numpy()[0]
        idx = int(np.argmax(preds))
        guess = categories[idx] if idx < len(categories) else "Unknown"

        return jsonify({
            "guess": guess,
            "confidence": float(preds[idx])
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
