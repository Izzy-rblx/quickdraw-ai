import os
import io
import base64
import hashlib
import numpy as np
from flask import Flask, request, jsonify
from tensorflow import keras
from PIL import Image, ImageDraw

# Flask app
app = Flask(__name__)

# Load categories
CATEGORIES_FILE = "categories.txt"
with open(CATEGORIES_FILE, "r") as f:
    categories = [line.strip() for line in f.readlines()]
print(f"[SERVER LOG] 📚 Loaded {len(categories)} categories.")

# Helper: log
def log(msg):
    print(f"[SERVER LOG] {msg}", flush=True)

# Helper: checksum
def sha256sum(filename):
    h = hashlib.sha256()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

# Load model
MODEL_PATH = os.getenv("QUICKDRAW_MODEL_PATH", "quickdraw_model.keras")
log(f"🔍 Looking for model at: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

log(f"📦 Found model file at {MODEL_PATH}, size: {os.path.getsize(MODEL_PATH)/1024:.2f} KB")
log(f"✅ Model checksum (sha256): {sha256sum(MODEL_PATH)}")

# If mislabeled as .keras but actually h5
if MODEL_PATH.endswith(".keras"):
    with open(MODEL_PATH, "rb") as f:
        header = f.read(8)
    if not (header.startswith(b"\x89HDF") or header.startswith(b"\x93NUMPY")):
        log("🔎 File has .keras extension but is NOT a zip; renaming to .h5")
        new_path = MODEL_PATH.replace(".keras", ".h5")
        os.rename(MODEL_PATH, new_path)
        MODEL_PATH = new_path

# Load Keras model
model = keras.models.load_model(MODEL_PATH)
log("✅ Model loaded successfully.")

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "QuickDraw AI API", "ok": True})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON received"}), 400

        img = None

        # ✅ Prefer strokes input (Roblox case)
        if "strokes" in data and isinstance(data["strokes"], list):
            log("🖊 Using strokes input")
            canvas = Image.new("L", (28, 28), 0)
            draw = ImageDraw.Draw(canvas)

            for stroke in data["strokes"]:
                points = [(int(p["x"] / 10), int(p["y"] / 10)) for p in stroke if "x" in p and "y" in p]
                if len(points) > 1:
                    draw.line(points, fill=255, width=1)
                elif points:
                    draw.point(points[0], fill=255)

            img = np.array(canvas).astype("float32") / 255.0
            img = np.expand_dims(img, axis=(0, -1))

        # ✅ Or base64 image input
        elif "image" in data and isinstance(data["image"], str):
            log("🖼 Using base64 image input")
            img_bytes = base64.b64decode(data["image"])
            img = Image.open(io.BytesIO(img_bytes)).convert("L").resize((28, 28))
            img = np.array(img).astype("float32") / 255.0
            img = np.expand_dims(img, axis=(0, -1))

        else:
            return jsonify({"error": "Invalid input format"}), 400

        # Run prediction
        preds = model.predict(img)
        idx = int(np.argmax(preds[0]))
        guess = categories[idx] if 0 <= idx < len(categories) else "unknown"

        log(f"🤖 Prediction: {guess} (index {idx})")
        return jsonify({"guess": guess})

    except Exception as e:
        log(f"❌ Error in /predict: {e}")
        return jsonify({"error": str(e)}), 500

# Run app
if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    log(f"🚀 Starting server on port {port}")
    app.run(host="0.0.0.0", port=port)
