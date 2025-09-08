import os
import sys
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

MODEL_PATH = "quickdraw_model.keras"

# Check if model file exists before loading
if not os.path.exists(MODEL_PATH):
    print(f"❌ ERROR: Model file '{MODEL_PATH}' not found. "
          f"Make sure it was downloaded in render-build.sh.", file=sys.stderr)
    sys.exit(1)  # Exit immediately, prevents crashing later

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Successfully loaded model from {MODEL_PATH}")
except Exception as e:
    print(f"❌ ERROR: Failed to load model from '{MODEL_PATH}': {e}", file=sys.stderr)
    sys.exit(1)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        image = Image.open(file).convert("L").resize((28, 28))  # grayscale + resize
        img_array = np.array(image).reshape(1, 28, 28, 1) / 255.0

        predictions = model.predict(img_array)
        predicted_class = int(np.argmax(predictions[0]))

        return jsonify({"class": predicted_class, "confidence": float(np.max(predictions[0]))})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return "✅ QuickDraw AI is running!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
