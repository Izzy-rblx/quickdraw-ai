import io
import os
import numpy as np
from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image

# Load categories
with open("categories.txt", "r") as f:
    categories = [line.strip() for line in f.readlines()]

# Use absolute path to model file
MODEL_PATH = os.path.join(os.path.dirname(__file__), "quickdraw_model.keras")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

app = Flask(__name__)

def preprocess_image(image_bytes):
    """Preprocess incoming image (28x28 grayscale like QuickDraw dataset)."""
    image = Image.open(io.BytesIO(image_bytes)).convert("L")  # grayscale
    image = image.resize((28, 28))  # match training size
    img_array = np.array(image).astype("float32") / 255.0
    img_array = 1.0 - img_array  # invert colors (QuickDraw is white bg, black strokes)
    img_array = np.expand_dims(img_array, axis=0)  # batch dimension
    img_array = np.expand_dims(img_array, axis=-1)  # channel dimension
    return img_array

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img_array = preprocess_image(file.read())
    preds = model.predict(img_array)
    class_id = int(np.argmax(preds[0]))
    confidence = float(np.max(preds[0]))

    return jsonify({
        "category": categories[class_id],
        "confidence": confidence
    })

if __name__ == "__main__":
    # Use Render’s PORT env variable if available, else default to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
