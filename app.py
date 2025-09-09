import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import keras

app = Flask(__name__)
CORS(app)

# Load categories
with open("categories.txt", "r") as f:
    categories = [line.strip() for line in f.readlines()]
print(f"✅ Loaded {len(categories)} categories")

# Load model (using TFSMLayer because it's a SavedModel)
model = None
try:
    model = keras.layers.TFSMLayer("quickdraw_saved_model", call_endpoint="serving_default")
    print("✅ Model loaded successfully (TFSMLayer)")
except Exception as e:
    print(f"⚠️ Failed to load model: {e}")

# --- Utility: Convert strokes → 28x28 numpy array ---
def strokes_to_bitmap(strokes, width=28, height=28):
    bitmap = np.zeros((height, width), dtype=np.float32)

    for stroke in strokes:
        for point in stroke:
            x = int(point.get("x", 0) / 10)
            y = int(point.get("y", 0) / 10)
            if 0 <= x < width and 0 <= y < height:
                bitmap[y, x] = 1.0  # white pixel

    return bitmap

@app.route("/")
def index():
    return "QuickDraw AI API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()
    if not data or "strokes" not in data:
        return jsonify({"error": "No strokes provided"}), 400

    try:
        # Convert strokes → bitmap
        bitmap = strokes_to_bitmap(data["strokes"], 28, 28)

        # Reshape for model: (1, 28, 28, 1)
        input_tensor = np.expand_dims(bitmap, axis=(0, -1)).astype(np.float32)

        # Run prediction
        preds = model(input_tensor, training=False).numpy()[0]
        best_idx = int(np.argmax(preds))
        guess = categories[best_idx]

        return jsonify({"guess": guess})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

