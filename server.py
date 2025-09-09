import os
import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image

app = Flask(__name__)

# Path to the model
MODEL_PATH = "quickdraw_model.keras"

# Debug: print absolute path info
abs_model_path = os.path.abspath(MODEL_PATH)
print(f"🔍 Looking for model at: {abs_model_path}")
print(f"📂 Current working directory: {os.getcwd()}")
print(f"📄 Files in CWD: {os.listdir(os.getcwd())}")

# Try loading the model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ ERROR: Failed to load model from {abs_model_path}: {e}")
    model = None


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    try:
        image = Image.open(file).convert("L").resize((28, 28))
        image_array = np.array(image) / 255.0
        image_array = image_array.reshape(1, 28, 28, 1)

        predictions = model.predict(image_array)
        predicted_class = int(np.argmax(predictions, axis=1)[0])

        return jsonify({"predicted_class": predicted_class})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
