from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import os

# Paths
MODEL_PATH = "quickdraw_saved_model"  # directory-based SavedModel
CATEGORIES_PATH = "categories.txt"

# Globals
model = None
CATEGORIES = []

# Try loading model
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully from SavedModel format")
    except Exception as e:
        print(f"⚠️ Failed to load model: {e}")
else:
    print(f"⚠️ Model path not found: {MODEL_PATH}")

# Try loading categories
if os.path.exists(CATEGORIES_PATH):
    with open(CATEGORIES_PATH, "r") as f:
        CATEGORIES = [line.strip() for line in f if line.strip()]
    print(f"✅ Loaded {len(CATEGORIES)} categories")
else:
    print("⚠️ categories.txt not found")

# Flask app
app = Flask(__name__)

def preprocess_strokes(strokes, size=28):
    """
    Convert strokes (list of points) into a 28x28 grayscale image.
    """
    bitmap = np.zeros((size, size), dtype=np.uint8)
    for stroke in strokes:
        for point in stroke:
            x = min(size - 1, max(0, int(point["x"] / 10)))
            y = min(size - 1, max(0, int(point["y"] / 10)))
            bitmap[y, x] = 255
    bitmap = bitmap.astype("float32") / 255.0
    bitmap = np.expand_dims(bitmap, axis=(0, -1))  # (1, 28, 28, 1)
    return bitmap

@app.route("/")
def home():
    return jsonify({"status": "ok", "message": "QuickDraw AI running!"})

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    try:
        data = request.get_json()
        strokes = data.get("image")

        if not strokes:
            return jsonify({"error": "No stroke data received"}), 400

        img = preprocess_strokes(strokes)
        preds = model.predict(img)
        idx = int(np.argmax(preds[0]))
        guess = CATEGORIES[idx] if idx < len(CATEGORIES) else "?"

        return jsonify({
            "guess": guess,
            "confidence": float(np.max(preds[0]))
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
