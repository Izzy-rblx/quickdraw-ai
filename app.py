import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU to save memory

from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import json

# Load categories
with open("categories.json", "r") as f:
    categories = json.load(f)

print(f"✅ Loaded {len(categories)} categories")

print("✅ Loading model...")
model = tf.saved_model.load("model")
print("✅ Model loaded successfully")

app = Flask(__name__)

@app.route("/")
def index():
    return "QuickDraw AI is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        if not data or "strokes" not in data:
            return jsonify({"error": "Missing strokes field"}), 400

        # Convert strokes to numpy format
        strokes = np.array(data["strokes"], dtype=np.float32)
        strokes = np.expand_dims(strokes, axis=0)

        # Run inference
        infer = model.signatures["serving_default"]
        preds = infer(tf.constant(strokes))["output_0"].numpy()[0]

        top_idx = int(np.argmax(preds))
        top_category = categories[top_idx]
        confidence = float(preds[top_idx])

        return jsonify({"category": top_category, "confidence": confidence})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
