import os
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np

app = Flask(__name__)

# Get absolute path to the model file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "quickdraw_model.keras")

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Load categories
CATEGORIES_PATH = os.path.join(BASE_DIR, "categories.txt")
with open(CATEGORIES_PATH, "r") as f:
    categories = [line.strip() for line in f.readlines()]

def preprocess_image(image_file):
    """Preprocess uploaded image for model prediction"""
    image = Image.open(image_file).convert("L")  # grayscale
    image = image.resize((28, 28))               # resize to match model
    image = np.array(image) / 255.0              # normalize to [0,1]
    image = image.reshape(1, 28, 28, 1)          # add batch & channel dim
    return image

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    image = preprocess_image(file)
    predictions = model.predict(image)
    predicted_index = np.argmax(predictions)
    predicted_label = categories[predicted_index]
    confidence = float(np.max(predictions))

    return jsonify({
        "prediction": predicted_label,
        "confidence": confidence
    })

@app.route("/")
def home():
    return "QuickDraw AI Model is running 🚀"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
