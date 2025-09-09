import os
import io
import hashlib
import logging
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import tensorflow as tf

# -----------------------------------------------------------------------------
# Flask setup
# -----------------------------------------------------------------------------
app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("QuickDrawServer")

# -----------------------------------------------------------------------------
# Model setup
# -----------------------------------------------------------------------------
MODEL_PATH = "quickdraw_model.keras"

if not os.path.exists(MODEL_PATH):
    logger.error(f"❌ Model file not found: {MODEL_PATH}")
    raise FileNotFoundError(f"{MODEL_PATH} missing, upload it first")

# Log file info
size = os.path.getsize(MODEL_PATH)
logger.info(f"📦 Model file size: {size/1024:.2f} KB")

with open(MODEL_PATH, "rb") as f:
    checksum = hashlib.sha256(f.read()).hexdigest()
logger.info(f"🔑 Model SHA256 checksum: {checksum}")

# Load model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("✅ Model loaded successfully")
except Exception as e:
    logger.exception("❌ Failed to load model")
    raise

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def preprocess_bitmap(bitmap):
    """Convert Roblox bitmap JSON → model input (28x28 grayscale)."""
    logger.info("🖼️ Preprocessing bitmap...")

    try:
        arr = np.array(bitmap, dtype=np.float32)
        if arr.shape != (28, 28):
            logger.warning(f"⚠️ Bitmap shape mismatch: got {arr.shape}, expected (28, 28)")
            return None
        arr = arr / 255.0  # normalize
        arr = arr.reshape(1, 28, 28, 1)
        return arr
    except Exception as e:
        logger.exception("❌ Failed during preprocessing")
        return None

def predict_image(arr):
    """Run model prediction and return best class."""
    try:
        preds = model.predict(arr)
        idx = np.argmax(preds[0])
        confidence = float(np.max(preds[0]))
        return idx, confidence
    except Exception as e:
        logger.exception("❌ Prediction failed")
        return None, None

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "QuickDraw AI API", "ok": True})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        logger.info(f"📩 Incoming JSON: {str(data)[:300]}")  # truncate long logs

        if not data or "image" not in data:
            logger.warning("⚠️ Missing 'image' in payload")
            return jsonify({"error": "Missing image"}), 400

        bitmap = data["image"]
        arr = preprocess_bitmap(bitmap)
        if arr is None:
            return jsonify({"error": "Invalid bitmap"}), 400

        idx, confidence = predict_image(arr)
        if idx is None:
            return jsonify({"error": "Prediction failed"}), 500

        result = {
            "guess": str(idx),
            "confidence": round(confidence, 4)
        }
        logger.info(f"✅ Prediction: {result}")
        return jsonify(result)

    except Exception as e:
        logger.exception("❌ Unexpected error in /predict")
        return jsonify({"error": "Server error"}), 500

# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"🚀 Starting server on port {port}")
    app.run(host="0.0.0.0", port=port)
