from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import json

# 🔹 Initialize Flask
app = Flask(__name__)
CORS(app)

# 🔹 Load model + categories
print("✅ Loading model...")
model = tf.saved_model.load("model")
with open("categories.json", "r") as f:
    categories = json.load(f)
print(f"✅ Loaded {len(categories)} categories")

@app.route("/")
def home():
    return "QuickDraw AI API is running ✅"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data or "strokes" not in data:
            return jsonify({"error": "No strokes provided"}), 400

        strokes = data["strokes"]

        # 🔹 Convert strokes → 28x28 bitmap
        img = np.zeros((28, 28), dtype=np.uint8)

        for stroke in strokes:
            for point in stroke:
                x = int(point.get("x", 0) / 10)
                y = int(point.get("y", 0) / 10)
                if 0 <= x < 28 and 0 <= y < 28:
                    img[y, x] = 255

        # Normalize to [0,1]
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=(0, -1))  # shape (1, 28, 28, 1)

        # 🔹 Predict
        preds = model(img, training=False).numpy()[0]
        top_idx = int(np.argmax(preds))
        guess = categories[top_idx]
        confidence = float(preds[top_idx])

        return jsonify({
            "guess": guess,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 🔹 Run locally (Render will use Gunicorn)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
