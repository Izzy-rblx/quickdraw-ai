import io
import base64
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image, ImageOps
import tensorflow as tf

# ----------------------
# Load QuickDraw model
# ----------------------
print("Loading QuickDraw model...")
model = tf.keras.models.load_model("quickdraw_model.h5", compile=False, safe_mode=False)
print("✅ Model loaded")

# ----------------------
# Load categories
# ----------------------
with open("categories.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]
print(f"✅ Loaded {len(class_names)} categories")

# ----------------------
# Flask app
# ----------------------
app = Flask(__name__)

def preprocess_image(image_data):
    """Convert base64 → PIL → 28x28 grayscale numpy array"""
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    image = ImageOps.grayscale(image)
    image = image.resize((28, 28))
    img_array = np.array(image).astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if "image" not in data:
            return jsonify({"error": "No image provided"}), 400

        img_array = preprocess_image(data["image"])

        # Run prediction
        prediction = model.predict(img_array)
        predicted_index = int(np.argmax(prediction))
        predicted_label = class_names[predicted_index]
        confidence = float(np.max(prediction))

        print("Prediction:", predicted_label, "(", confidence, ")")

        return jsonify({
            "label": predicted_label,
            "confidence": confidence
        })
    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
