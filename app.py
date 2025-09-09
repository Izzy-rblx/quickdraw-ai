from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

# ✅ Create Flask app
app = Flask(__name__)

# ✅ Load categories
with open("categories.txt", "r") as f:
    categories = [line.strip() for line in f.readlines() if line.strip()]

print(f"✅ Loaded {len(categories)} categories")

# ✅ Load the SavedModel (use the folder you uploaded)
print("✅ Loading model...")
model = tf.saved_model.load("quickdraw_saved_model")
infer = model.signatures["serving_default"]
print("✅ Model loaded successfully")

# 🔹 Convert strokes (28x28 int array) into float32 normalized tensor
def preprocess_bitmap(bitmap):
    arr = np.array(bitmap, dtype=np.float32) / 255.0  # normalize 0–1
    arr = arr.reshape(1, 28, 28, 1)  # match model input shape
    return tf.convert_to_tensor(arr, dtype=tf.float32)

@app.route("/", methods=["GET"])
def home():
    return "✅ QuickDraw AI API is running!", 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No image data provided"}), 400

        # 🔹 Preprocess input
        bitmap = data["image"]
        input_tensor = preprocess_bitmap(bitmap)

        # 🔹 Run inference
        outputs = infer(tf.constant(input_tensor))
        probs = list(outputs.values())[0].numpy()[0]

        # 🔹 Pick top prediction
        top_idx = int(np.argmax(probs))
        guess = categories[top_idx]
        confidence = float(probs[top_idx])

        return jsonify({
            "guess": guess,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        print("❌ Error in /predict:", str(e))
        return jsonify({"error": str(e)}), 500

# ✅ Render/Gunicorn entrypoint
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
