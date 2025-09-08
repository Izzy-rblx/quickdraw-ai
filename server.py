import io
import base64
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
import tensorflow as tf

# Load the legacy QuickDraw model (.h5 format)
print("Loading QuickDraw model...")
print("Loading QuickDraw model...")
model = tf.keras.models.load_model("quickdraw_model.keras", compile=False)
print("Model loaded successfully!")
print("Model loaded successfully!")

# Your class labels – must match the trained dataset
# Replace with the actual QuickDraw classes your model supports
class_names = [
    "cat", "dog", "car", "house", "tree", "bicycle", "airplane", "fish",
    "flower", "clock", "star", "sun", "moon", "shoe", "cup"
]

app = Flask(__name__)

def preprocess_image(image_bytes):
    """Convert incoming image bytes to model-ready tensor"""
    img = Image.open(io.BytesIO(image_bytes)).convert("L")  # grayscale
    img = img.resize((28, 28))  # QuickDraw models usually use 28x28
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # add channel
    img_array = np.expand_dims(img_array, axis=0)   # batch dim
    return img_array

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if "image" not in data:
        return jsonify({"error": "No image field"}), 400

    try:
        # Decode base64 image
        image_data = base64.b64decode(data["image"])
        input_tensor = preprocess_image(image_data)

        # Predict
        preds = model.predict(input_tensor)
        top_idx = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0]))

        return jsonify({
            "prediction": class_names[top_idx],
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return "QuickDraw AI server is running!"

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
