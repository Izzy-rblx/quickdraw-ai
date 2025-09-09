from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import os
import traceback

# === Load model ===
MODEL_PATH = "model.h5"  # or quickdraw_saved_model if you exported differently
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

# === Load categories from file ===
CATEGORIES_FILE = "categories.txt"
if not os.path.exists(CATEGORIES_FILE):
    raise FileNotFoundError(f"Categories file not found at {CATEGORIES_FILE}")
with open(CATEGORIES_FILE, "r", encoding="utf-8") as f:
    CATEGORIES = [line.strip() for line in f if line.strip()]

app = Flask(__name__)

# === Preprocess strokes into a 28x28 image ===
def preprocess_strokes(strokes, size=28):
    bitmap = np.zeros((size, size), dtype=np.uint8)
    for stroke in strokes:
        for point in stroke:
            x = min(size - 1, max(0, int(point["x"] / 10)))
            y = min(size - 1, max(0, int(point["y"] / 10)))
            bitmap[y, x] = 255
    bitmap = bitmap.astype("float32") / 255.0
    # Add channel dimension if needed (28,28,1)
    bitmap = np.expand_dims(bitmap, axis=(0, -1))
    return bitmap

@app.route("/")
def home():
    return jsonify({"status": "ok", "message": "QuickDraw AI running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        strokes = data.get("image")

        if not strokes:
            return jsonify({"error": "No stroke data received"}), 400

        img = preprocess_strokes(strokes)

        preds = model(img).numpy()
        idx = int(np.argmax(preds[0]))
        guess = CATEGORIES[idx] if idx < len(CATEGORIES) else "?"

        return jsonify({
            "guess": guess,
            "confidence": float(np.max(preds[0]))
        })

    except Exception as e:
        traceback.print_exc()  # log full error in Render logs
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # For local testing
    app.run(host="0.0.0.0", port=7860)
