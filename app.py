import os
import json
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify

app = Flask(__name__)

# --- Step 1: Ensure categories.json exists ---
if not os.path.exists("categories.json"):
    print("⚠️ categories.json not found, generating from categories.txt...")
    if os.path.exists("categories.txt"):
        with open("categories.txt", "r") as f:
            categories = [line.strip() for line in f if line.strip()]
        with open("categories.json", "w") as f:
            json.dump(categories, f, indent=2)
        print(f"✅ Generated categories.json with {len(categories)} items")
    else:
        raise FileNotFoundError("❌ categories.txt is missing! Please add it to your repo.")

# --- Step 2: Load categories ---
with open("categories.json", "r") as f:
    categories = json.load(f)
print(f"✅ Loaded {len(categories)} categories")

# --- Step 3: Load model ---
print("✅ Loading model...")
model = tf.saved_model.load("model")
infer = model.signatures["serving_default"]
print("✅ Model loaded successfully")

# --- Step 4: Prediction route ---
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        if not data or "strokes" not in data:
            return jsonify({"error": "Invalid input, expected JSON with 'strokes'"}), 400

        strokes = np.array(data["strokes"], dtype=np.float32)
        strokes = np.expand_dims(strokes, axis=0)  # batch size 1

        outputs = infer(tf.constant(strokes))
        predictions = list(outputs.values())[0].numpy()[0]

        top_idx = int(np.argmax(predictions))
        confidence = float(predictions[top_idx])
        category = categories[top_idx]

        return jsonify({
            "prediction": category,
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Step 5: Health check ---
@app.route("/", methods=["GET"])
def health():
    return "✅ QuickDraw AI is running!", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
