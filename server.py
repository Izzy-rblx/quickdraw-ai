import os
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

MODEL_PATH = "quickdraw_model.keras"
model = None

# Try loading the model
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"✅ Successfully loaded model from {MODEL_PATH}")
    except Exception as e:
        print(f"❌ ERROR: Failed to load model from '{MODEL_PATH}': {e}")
        model = None
else:
    print(f"❌ ERROR: Model file '{MODEL_PATH}' not found. Did render-build.sh download it?")

# Load categories (labels)
CATEGORIES_FILE = "categories.txt"
categories = []
if os.path.exists(CATEGORIES_FILE):
    with open(CATEGORIES_FILE, "r") as f:
        categories = [line.strip() for line in f.readlines()]
    print(f"✅ Loaded {len(categories)} categories from {CATEGORIES_FILE}")
else:
    print(f"⚠️ WARNING: categories.txt not found, predictions will use index numbers.")

def preprocess_image(image):
    """Resize and normalize input image for the model"""
    image = image.convert("L")              # grayscale
    image = image.resize((28, 28))          # resize to 28x28
    img_array = np.array(image) / 255.0     # normalize to [0, 1]
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

@app.route("/")
def home():
    return jsonify({
        "status": "ok",
        "message": "QuickDraw AI server is running.",
        "model_loaded": model is not None
    })

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded on server"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    try:
        image = Image.open(file.stream)
        processed = preprocess_image(image)
        preds = model.predict(processed)[0]
        top_idx = int(np.argmax(preds))
        confidence = float(preds[top_idx])

        label = categories[top_idx] if categories and top_idx < len(categories) else str(top_idx)

        return jsonify({
            "prediction": label,
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
