import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

# 🔹 Flask setup
app = Flask(__name__)
CORS(app)

# 🔹 Load categories
with open("categories.txt", "r") as f:
    categories = [line.strip() for line in f.readlines()]
print(f"✅ Loaded {len(categories)} categories")

# 🔹 Load model
print("✅ Loading model...")
model = tf.keras.layers.TFSMLayer("quickdraw_saved_model", call_endpoint="serving_default")
print("✅ Model loaded successfully")

@app.route("/")
def home():
    return "QuickDraw AI is running!", 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 🔹 Log raw body for debugging
        raw_data = request.data.decode("utf-8")
        print("📥 Raw Request Body (first 500 chars):", raw_data[:500])

        # Parse JSON
        data = request.get_json(force=True)
        print("📥 Parsed JSON keys:", list(data.keys()))

        if "image" not in data:
            return jsonify({"error": "Missing 'image' field"}), 400

        # Convert to numpy
        image = np.array(data["image"], dtype=np.float32).reshape(1, 28, 28, 1) / 255.0
        print("✅ Image shape:", image.shape, "dtype:", image.dtype)

        # Predict
        preds = model(image)[0].numpy()
        guess_idx = int(np.argmax(preds))
        guess = categories[guess_idx]

        print("🤖 Guess:", guess)

        return jsonify({"guess": guess}), 200

    except Exception as e:
        print("❌ Error in /predict:", str(e))
        return jsonify({"error": str(e)}), 500

# 🔹 Entry point for Render/Gunicorn
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
