import os
import io
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import tensorflow as tf
import logging

# ----------------- Logging -----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("QuickDrawServer")

# ----------------- Flask App -----------------
app = Flask(__name__)

# ----------------- Model Loading -----------------
MODEL_PATH = None
model = None

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
    logger.info(f"📦 Loading SavedModel with TFSMLayer: {MODEL_PATH}")
    model = tf.keras.layers.TFSMLayer(MODEL_PATH, call_endpoint="serving_default")

else:
    logger.error("❌ No model file found! Please upload .keras or .h5")
    raise FileNotFoundError("No valid model found")

# ----------------- Categories -----------------
# Load categories from file
if os.path.exists("categories.txt"):
    with open("categories.txt", "r") as f:
        categories = [line.strip() for line in f.readlines()]
    logger.info(f"✅ Loaded {len(categories)} categories.")
else:
    categories = [f"class_{i}" for i in range(model.output_shape[-1])]
    logger.warning("⚠️ categories.txt not found, using placeholder classes.")

# ----------------- Preprocessing -----------------
def preprocess_image(image_bytes):
    """Convert input bytes into a model-ready array"""
    image = Image.open(io.BytesIO(image_bytes)).convert("L").resize((28, 28))
    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))  # (1, 28, 28, 1)
    return img_array

# ----------------- Routes -----------------
@app.route("/")
def home():
    return jsonify({"status": "ok", "message": "QuickDraw AI server is running 🚀"})

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    img_array = preprocess_image(file.read())

    try:
        preds = model(img_array)
        if isinstance(preds, dict):  # if using TFSMLayer
            preds = preds["outputs"]
        preds = preds.numpy()[0]

        top_idx = int(np.argmax(preds))
        top_label = categories[top_idx] if top_idx < len(categories) else f"class_{top_idx}"
        confidence = float(preds[top_idx])

        return jsonify({"prediction": top_label, "confidence": confidence})

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

# ----------------- Run -----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
