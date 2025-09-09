import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

app = Flask(__name__)

MODEL_PATH = "model.h5"
CATEGORIES_PATH = "categories.txt"

# Try loading model
model = None
categories = []

if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"⚠️ Failed to load model: {e}")
else:
    print(f"⚠️ Model file not found at {MODEL_PATH}. The API will run, but predictions won't work.")

# Load categories if available
if os.path.exists(CATEGORIES_PATH):
    with open(CATEGORIES_PATH, "r") as f:
        categories = [line.strip() for line in f.readlines()]
    print(f"✅ Loaded {len(categories)} categories")
else:
    print(f"⚠️ Categories file not found at {CATEGORIES_PATH}")

@app.route("/", methods=["GET"])
def home():
    return "QuickDraw AI is running! 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or not categories:
        return jsonify({"error": "Model or categories not available on server"}), 503

    try:
        # Read image from request
        file = request.files["file"]
        image = Image.open(file).convert("L").resize((28, 28))
        image = img_to_array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        # Make prediction
        preds = model.predict(image)[0]
        top_index = np.argmax(preds)
        top_category = categories[top_index]
        confidence = float(preds[top_index])

        return jsonify({
            "prediction": top_category,
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
