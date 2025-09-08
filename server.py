from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load model once at startup
print("Loading QuickDraw model...")
model = tf.keras.models.load_model("quickdraw_model.h5")

# Load categories
with open("categories.txt") as f:
    categories = [line.strip() for line in f]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    strokes = data.get("strokes", [])

    # Create 28x28 empty image
    img = np.zeros((28, 28), dtype=np.float32)

    # Scale strokes down to 28x28
    for stroke in strokes:
        for point in stroke:
            x = int(point["x"] / 10)  # scale (Roblox canvas ~280x280)
            y = int(point["y"] / 10)
            if 0 <= x < 28 and 0 <= y < 28:
                img[y, x] = 1.0

    # Reshape for model
    img = img.reshape(1, 28, 28, 1)

    # Predict
    preds = model.predict(img, verbose=0)
    idx = int(np.argmax(preds))
    guess = categories[idx]

    return jsonify({"guess": guess})

