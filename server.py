import os
import io
import base64
import hashlib
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import tensorflow as tf

# -------------------- CONFIG --------------------
MODEL_PATH = os.environ.get("QUICKDRAW_MODEL_PATH", "quickdraw_model.keras")
CATEGORIES_FILE = os.environ.get("CATEGORIES_FILE", "categories.txt")

app = Flask(__name__)

# -------------------- LOGGING UTILS --------------------
def log(msg):
    print(f"[SERVER LOG] {msg}", flush=True)

def file_sha256(path):
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

# -------------------- LOAD CATEGORIES --------------------
if os.path.exists(CATEGORIES_FILE):
    with open(CATEGORIES_FILE, "r") as f:
        categories = [line.strip() for line in f.readlines()]
    log(f"📚 Loaded {len(categories)} categories.")
else:
    categories = []
    log("⚠️ No categories.txt found!")

# -------------------- LOAD MODEL --------------------
log(f"🔍 Looking for model at: {os.path.abspath(MODEL_PATH)}")
log(f"📂 Current working directory: {os.getcwd()}")
log(f"📄 Files in CWD: {os.listdir('.')}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found at {MODEL_PATH}")

size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
log(f"📦 Found model file {MODEL_PATH}, size: {size:.2f} MB")
log(f"✅ Model checksum (sha256): {file_sha256(MODEL_PATH)}")

# Handle wrong extension
model_file_to_load = MODEL_PATH
if MODEL_PATH.endswith(".keras"):
    with open(MODEL_PATH, "rb") as f:
        header = f.read(4)
    if header != b"\x93HDF":
        log("🔎 File has .keras extension but is NOT zip; renaming to .h5...")
        h5_path = MODEL_PATH.replace(".keras", ".h5")
        os.rename(MODEL_PATH, h5_path)
        model_file_to_load = h5_path

# Load
log("📚 Loading model...")
model = tf.keras.models.load_model(model_file_to_load)
log("✅ Model loaded successfully.")

# Show input shape
try:
    log(f"🧪 Model inputs: {model.inputs}")
except Exception:
    log("⚠️ Could not inspect model inputs.")

# -------------------- ROUTES --------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "QuickDraw AI API", "ok": True})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON received"}), 400

        # Case 1: strokes (from Roblox)
        if "strokes" in data:
            log("🖊 Received strokes input")
            strokes = data["strokes"]
            canvas = np.zeros((28, 28), dtype=np.uint8)

            # normalize strokes into 28x28
            for stroke in strokes:
                for point in stroke:
                    x = int(point["x"] / 10)  # ⚠ adjust divisor depending on UI size
                    y = int(point["y"] / 10)
                    if 0 <= x < 28 and 0 <= y < 28:
                        canvas[y, x] = 255

            img = canvas.astype("float32") / 255.0
            img = np.expand_dims(img, axis=(0, -1))

        # Case 2: base64 image
        elif "image" in data:
            log("🖼 Received image input")
            img_bytes = base64.b64decode(data["image"])
            img = Image.open(io.BytesIO(img_bytes)).convert("L").resize((28, 28))
            img = np.array(img).astype("float32") / 255.0
            img = np.expand_dims(img, axis=(0, -1))

        else:
            return jsonify({"error": "No strokes or image provided"}), 400

        # Run prediction
        preds = model.predict(img)
        idx = int(np.argmax(preds[0]))
        guess = categories[idx] if 0 <= idx < len(categories) else "unknown"

        log(f"🤖 Prediction: {guess} (index {idx})")
        return jsonify({"guess": guess})

    except Exception as e:
        log(f"❌ Error in /predict: {e}")
        return jsonify({"error": str(e)}), 500

# -------------------- MAIN --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    log("🚀 Starting server...")
    log(f"🧭 ENV PORT={port}")
    log(f"🧭 ENV QUICKDRAW_MODEL_PATH={MODEL_PATH}")
    log(f"🧭 CATEGORIES_FILE={CATEGORIES_FILE}")
    app.run(host="0.0.0.0", port=port)
