import os
import logging
import hashlib
import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image

# ---------------------------------------------------------------------
# Setup logging
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("QuickDrawServer")

# ---------------------------------------------------------------------
# Model + categories setup
# ---------------------------------------------------------------------
MODEL_PATH = os.getenv("QUICKDRAW_MODEL_PATH", "quickdraw_model.h5")  # ✅ now uses .h5
CATEGORIES_FILE = os.getenv("CATEGORIES_FILE", "categories.txt")

# Log working directory and files
logger.info("📂 Current working directory: %s", os.getcwd())
logger.info("📄 Files in CWD: %s", os.listdir(os.getcwd()))

# Check if model file exists
if not os.path.exists(MODEL_PATH):
    logger.error("❌ Model file not found at %s", MODEL_PATH)
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# Log size and checksum
file_size = os.path.getsize(MODEL_PATH) / 1024
logger.info("📦 Model file size: %.2f KB", file_size)

sha256_hash = hashlib.sha256()
with open(MODEL_PATH, "rb") as f:
    for byte_block in iter(lambda: f.read(4096), b""):
        sha256_hash.update(byte_block)
checksum = sha256_hash.hexdigest()
logger.info("🔑 Model SHA256 checksum: %s", checksum)

# Load model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("✅ Model loaded successfully.")
except Exception as e:
    logger.error("❌ Failed to load model", exc_info=True)
    raise e

# Load categories
categories = []
if os.path.exists(CATEGORIES_FILE):
    with open(CATEGORIES_FILE, "r") as f:
        categories = [line.strip() for line in f.readlines()]
    logger.info("📚 Loaded %d categories.", len(categories))
else:
    logger.warning("⚠️ Categories file not found (%s).", CATEGORIES_FILE)

# ---------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------
app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "QuickDraw AI API", "ok": True})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "Missing 'image' in request"}), 400

        # Expecting 28x28 grayscale bitmap (list of lists)
        bitmap = np.array(data["image"], dtype=np.uint8)
        if bitmap.shape != (28, 28):
            return jsonify({"error": f"Invalid image shape {bitmap.shape}, expected (28,28)"}), 400

        # Normalize + reshape
        img = bitmap.astype("float32") / 255.0
        img = np.expand_dims(img, axis=(0, -1))  # shape (1, 28, 28, 1)

        # Predict
        preds = model.predict(img)
        pred_idx = int(np.argmax(preds))
        confidence = float(np.max(preds))

        guess = categories[pred_idx] if categories and pred_idx < len(categories) else str(pred_idx)

        return jsonify({
            "guess": guess,
            "confidence": confidence
        })

    except Exception as e:
        logger.error("Prediction error", exc_info=True)
        return jsonify({"error": str(e)}), 500

# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    logger.info("🚀 Starting server on port %s", port)
    app.run(host="0.0.0.0", port=port)
