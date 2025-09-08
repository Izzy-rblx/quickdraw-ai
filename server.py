import io
import numpy as np
from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image

# Load categories
with open("categories.txt", "r") as f:
    categories = [line.strip() for line in f.readlines()]

# Load Keras model
model = tf.keras.models.load_model("quickdraw_model.keras", compile=False)

app = Flask(__name__)

def preprocess_image(image_bytes):
    """Preprocess incoming image (28x28 grayscale like QuickDraw dataset)."""
    image = Image.open(io.BytesIO(image_bytes)).convert("L")  # grayscale
    image = image.resize((28, 28))  # match training size
    img_array = np.array(image).astype("float32") / 255.0
    img_array = 1.0 - img_array  # invert colors (QuickDraw is white background, black strokes)
    img_array = np.expand_dims(img_array, axis=0)  # batch dimension
    img_array = np.expand_dims(img_array, axis=-1) # channel dimension
    return img_array

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    img_array = preprocess_image(file.read())
    preds = model.predict(img_array)[0]

    top_idx = np.argmax(preds)
    top_label = categories[top_idx]
    top_conf = float(preds[top_idx])

    return jsonify({
        "prediction": top_label,
        "confidence": round(top_conf, 4)
    })

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

