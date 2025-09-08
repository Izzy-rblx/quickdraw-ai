import os
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import tensorflow as tf

# Load model (make sure quickdraw_model.keras is in the same folder)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "quickdraw_model.keras")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Load categories
with open(os.path.join(os.path.dirname(__file__), "categories.txt"), "r") as f:
    categories = [line.strip() for line in f]

# Initialize Flask app
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "✅ QuickDraw AI is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        img = Image.open(file).convert("L")  # grayscale
        img = img.resize((28, 28))           # match training input
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        preds = model.predict(img_array)
        pred_index = np.argmax(preds[0])
        pred_label = categories[pred_index]

        return jsonify({
            "prediction": pred_label,
            "confidence": float(np.max(preds[0]))
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ✅ Only needed when testing locally (not used on Render)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
