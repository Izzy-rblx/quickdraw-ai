from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

# ✅ Correct model file name
MODEL_PATH = "quickdraw_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# ✅ Categories (can be expanded / loaded from categories.txt if you prefer)
CATEGORIES = [
    "apple", "banana", "car", "cat", "dog", "house", "tree"
]

app = Flask(__name__)

def preprocess_strokes(strokes, size=28):
    """
    Convert stroke data from Roblox into a bitmap for the model.
    """
    bitmap = np.zeros((size, size), dtype=np.uint8)
    for stroke in strokes:
        for point in stroke:
            x = min(size - 1, max(0, int(point["x"] / 10)))
            y = min(size - 1, max(0, int(point["y"] / 10)))
            bitmap[y, x] = 255
    bitmap = bitmap.astype("float32") / 255.0
    bitmap = np.expand_dims(bitmap, axis=(0, -1))
    return bitmap

@app.route("/")
def home():
    return jsonify({"status": "ok", "message": "QuickDraw AI running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        strokes = data.get("image")

        if not strokes:
            return jsonify({"error": "No stroke data received"}), 400

        img = preprocess_strokes(strokes)

        preds = model.predict(img)
        idx = int(np.argmax(preds[0]))
        guess = CATEGORIES[idx] if idx < len(CATEGORIES) else "?"

        return jsonify({
            "guess": guess,
            "confidence": float(np.max(preds[0]))
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Hugging Face Spaces runs on port 7860
    app.run(host="0.0.0.0", port=7860)
