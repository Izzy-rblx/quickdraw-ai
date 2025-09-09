import os
import io
import hashlib
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image

# ==============================
# Configuration
# ==============================
MODEL_PATH = os.getenv("QUICKDRAW_MODEL_PATH", "quickdraw_model.keras")
CATEGORIES_FILE = os.getenv("CATEGORIES_FILE", "categories.txt")
PORT = int(os.getenv("PORT", 5000))

# ==============================
# Helper functions
# ==============================
def sha256_checksum(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(4096), b""):
            sha256.update(block)
    return sha256.hexdigest()

def preprocess_image(image_bytes):
    """Convert uploaded image bytes into (28,28,1) grayscale numpy array"""
    img = Image.open(io.BytesIO(image_bytes)).convert("L").resize((28, 28))
    img = np.array(img).astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)  # shape (28,28,1)
    img = np.expand_dims(img, axis=0)   # shape (1,28,28,1)
    return img

# ==============================
# Load model & categories
# ==============================
print("[SERVER LOG] 🚀 Starting server...")
print(f"[SERVER LOG] 🧭 ENV PORT={PORT}")
print(f"[SERVER LOG] 🧭 ENV QUICKDRAW_MODEL_PATH={MODEL_PATH}")
print(f"[SERVER LOG] 🧭 CATEGORIES_FILE={CATEGORIES_FILE}")

categories = []
if os.path.exists(CATEGORIES_FILE):
    with open(CATEGORIES_FILE, "r") as f:
        categories = [line.strip() for line in f.readlines()]
    print(f"[SERVER LOG] 📚 Loaded {len(categories)} categories.")
else:
    print("[SERVER LOG] ⚠️ Categories file not found!")
    categories = []

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

print(f"[SERVER LOG] 📦 Found model file at {MODEL_PATH}, size: {os.path.getsize(MODEL_PATH)/1e6:.2f} MB")
print(f"[SERVER LOG] ✅ Model checksum (sha256): {sha256_checksum(MODEL_PATH)}")

# Fix extension confusion between .keras and .h5
if MODEL_PATH.endswith(".keras"):
    with open(MODEL_PATH, "rb") as f:
        magic = f.read(4)
    if magic[:2] != b"\x50\x4B":  # not a zip -> HDF5
        print("[SERVER LOG] 🔎 File has .keras extension but is NOT a zip; likely an HDF5 model renamed.")
        new_path = MODEL_PATH.replace(".keras", ".h5")
        os.rename(MODEL_PATH, new_path)
        MODEL_PATH = new_path
        print(f"[SERVER LOG] ✏️  Renamed to {MODEL_PATH} for proper loading.")

print("[SERVER LOG] 📚 Loading model...")
model = load_model(MODEL_PATH)
print("[SERVER LOG] ✅ Model loaded successfully.")
print(f"[SERVER LOG] 🧪 Model inputs: {model.inputs}")
print("[SERVER LOG] ✅ Model ready for inference.")

# ==============================
# Flask API
# ==============================
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "QuickDraw AI API", "ok": True})

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "No file provided. Send as form-data with key 'file'."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"ok": False, "error": "Empty filename."}), 400

    try:
        img = preprocess_image(file.read())
        preds = model.predict(img)
        idx = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0]))
        label = categories[idx] if idx < len(categories) else str(idx)

        return jsonify({
            "ok": True,
            "prediction": label,
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# ==============================
# Run server
# ==============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
