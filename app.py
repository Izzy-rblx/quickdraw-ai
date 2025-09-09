import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

# Load the TensorFlow SavedModel
MODEL_PATH = "quickdraw_saved_model"
model = load_model(MODEL_PATH)

# Load categories
with open("categories.txt", "r") as f:
    categories = [line.strip() for line in f.readlines()]

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok", "message": "QuickDraw AI API is running 🚀"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No image data"}), 400

        # Convert JSON → numpy array
        img_array = np.array(data["image"], dtype=np.float32)

        # Ensure it's 28x28
        if img_array.shape != (28, 28):
            return jsonify({"error": f"Invalid image shape {img_array.shape}, expected (28,28)"}), 400

        # Normalize & reshape
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=(0, -1))  # shape (1,28,28,1)

        # Predict
        preds = model.predict(img_array)
        class_index = int(np.argmax(preds, axis=1)[0])
        confidence = float(np.max(preds))

        return jsonify({
            "guess": categories[class_index],
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
