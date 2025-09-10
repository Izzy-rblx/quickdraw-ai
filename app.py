from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import json

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("quickdraw_model.h5")

# Load QuickDraw label list (same order as model training)
with open("categories.json", "r") as f:
    CATEGORIES = json.load(f)

@app.route("/categories", methods=["GET"])
def get_categories():
    """Return all valid QuickDraw categories."""
    return jsonify(CATEGORIES)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    strokes = np.array(data["strokes"], dtype=np.float32)

    # Preprocess strokes into model input (dummy reshape, adjust if needed)
    strokes = strokes.reshape(1, 28, 28, 1) / 255.0  

    pred = model.predict(strokes)
    idx = int(np.argmax(pred))
    guess = CATEGORIES[idx]

    return jsonify({"guess": guess})
    
if __name__ == "__main__":
    app.run(port=5000, debug=True)
