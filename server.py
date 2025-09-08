from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load Google’s pretrained QuickDraw model (must be in your repo or downloaded)
model = tf.keras.models.load_model("quickdraw_model.h5")

# List of 345 QuickDraw classes (must match your model training)
with open("categories.txt") as f:
    categories = [line.strip() for line in f]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    strokes = data.get("strokes", [])

    # Convert strokes to image (28x28 bitmap like QuickDraw dataset)
    img = np.zeros((28, 28), dtype=np.float32)
    for stroke in strokes:
        for point in stroke:
            x = int(point["x"] / 10)  # scale down
            y = int(point["y"] / 10)
            if 0 <= x < 28 and 0 <= y < 28:
                img[y, x] = 1.0

    img = img.reshape(1, 28, 28, 1)

    # Predict
    preds = model.predict(img)
    idx = np.argmax(preds)
    guess = categories[idx]

    return jsonify({"guess": guess})
