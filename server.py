import os
import io
import logging
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import tensorflow as tf

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("QuickDrawServer")

# Initialize Flask
app = Flask(__name__)

# Load categories
with open("categories.txt", "r") as f:
    CATEGORIES = [line.strip() for line in f.readlines()]

# Try to load model (Keras v3 / H5 / SavedModel)
MODEL_PATH = None
model = None

logger.info("📂 Current directory: %s", os.getcwd())
logger.info("📄 Files here: %s", os.listdir(os.getcwd()))

if os.path.exists("quickdraw_model.keras"):
    MODEL_PATH = "quickdraw_model.keras"
    logger.info(f"📦 Loading Keras v3 model: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

elif os.path.exists("quickdraw_model.h5"):
    MODEL_PATH = "quickdraw_model.h5"
    logger.info(f"📦 Loading legacy H5 model: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

elif os.path.exists("quickdraw_saved_model"):
    MODEL_PATH = "quickdraw_saved_model"
    logger.info(f"📦 Loading TensorFlow SavedModel: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

else:
    raise FileNotFoundError("❌ No model file found (expected .keras, .h5, or SavedModel folder).")

logger.info("✅ Model loaded successfully!")


def preprocess_image(image_bytes):
    """Convert raw image bytes into model input tensor."""
    image = Image.open(io.BytesIO(image_bytes)).convert("L").resize((28, 28))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1).astype("float32")
    return img_array


@app.route("/predict", methods=["POST"])
def predict():
    """Handle prediction requests."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    img_array = preprocess_image(file.read())
    preds = model.predict(img_array)[0]

    top_idx = np.argmax(preds)
    result = {
        "category": CATEGORIES[top_idx],
        "confidence": float(preds[top_idx]),
    }
    return jsonify(result)


@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "categories": len(CATEGORIES)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
