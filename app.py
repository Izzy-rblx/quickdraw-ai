from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from keras.layers import TFSMLayer

MODEL_PATH_DIR = "quickdraw_saved_model"
model = TFSMLayer(MODEL_PATH_DIR, call_endpoint="serving_default")

# Load categories dynamically
with open("categories.txt", "r") as f:
    CATEGORIES = [line.strip() for line in f if line.strip()]

app = Flask(__name__)

def preprocess_strokes(strokes, size=28):
    bitmap = np.zeros((size, size), dtype=np.uint8)
    for stroke in strokes:
        for point in stroke:
            x = min(size - 1, max(0, int(point["x"] / 10)))
            y = min(size - 1, max(0, int(point["y"] / 10)))
            bitmap[y, x] = 255
    bitmap = bitmap.astype("float32") / 255.0
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

        return jsonify({"guess": guess, "confidence": float(np.max(preds[0]))})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
