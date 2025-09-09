# server.py
import os
import io
import sys
import json
import time
import hashlib
import zipfile
from pathlib import Path
from typing import Optional, Tuple

from flask import Flask, jsonify, request, abort

import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Logging helpers
# -----------------------------
def log(msg: str):
    print(f"[SERVER LOG] {msg}", flush=True)

def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def human_size(num_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"

# -----------------------------
# App setup
# -----------------------------
app = Flask(__name__)

# Environment / paths
DEFAULT_MODEL_PATH = "quickdraw_model.keras"  # what’s in your repo
MODEL_PATH_ENV = os.getenv("QUICKDRAW_MODEL_PATH", DEFAULT_MODEL_PATH)
MODEL_PATH = Path(MODEL_PATH_ENV)

CATEGORIES_PATH = Path(os.getenv("CATEGORIES_FILE", "categories.txt"))
PORT = int(os.getenv("PORT", "10000"))

# Global model & categories
model = None
CATEGORIES = []

# -----------------------------
# Utility: robust model loader
# -----------------------------
def is_keras_zip(path: Path) -> bool:
    # True if it looks like a valid ZIP (the .keras format is a zip file)
    try:
        return zipfile.is_zipfile(path)
    except Exception:
        return False

def _load_h5_legacy(path: Path):
    """
    Force-load an HDF5 model even if the file extension isn't .h5/.hdf5.
    Works around Keras 3's extension-based loader selection.
    """
    log("📚 Using legacy HDF5 loader path...")
    try:
        # Try the standard loader first (it will use HDF5 if extension is .h5/.hdf5)
        return tf.keras.models.load_model(str(path), compile=False)
    except Exception as e1:
        log(f"⚠️ Standard loader failed on HDF5 path: {type(e1).__name__}: {e1}")
        # Fall back to keras.saving.legacy.hdf5_format
        try:
            import h5py  # requirements.txt includes h5py
            try:
                from keras.saving.legacy.hdf5_format import load_model_from_hdf5
            except Exception:
                # Some TF/Keras builds expose it under tf.keras
                from tensorflow.keras.saving.hdf5_format import load_model_from_hdf5  # type: ignore

            with h5py.File(str(path), "r") as f:
                return load_model_from_hdf5(f)
        except Exception as e2:
            raise RuntimeError(
                f"Legacy HDF5 load failed: {type(e2).__name__}: {e2}"
            ) from e2

def load_model_robust(path: Path):
    """
    Load a model from .keras or .h5. If the file has a .keras extension but is NOT a zip,
    assume it's actually an HDF5 file that was renamed; transparently rename and load.
    """
    log(f"🔍 Looking for model at: {path.resolve()}")
    log(f"📂 Current working directory: {Path.cwd().resolve()}")
    try:
        files = sorted([p.name for p in Path.cwd().iterdir()])
        log(f"📄 Files in CWD: {files}")
    except Exception as e:
        log(f"⚠️ Could not list CWD: {e}")

    if not path.exists():
        raise FileNotFoundError(
            f"Model file does not exist: {path.resolve()}"
        )

    size = path.stat().st_size
    log(f"📦 Found model file at {path.name}, size: {human_size(size)}")
    checksum = sha256_of_file(path)
    log(f"✅ Model checksum (sha256): {checksum}")

    suffix = path.suffix.lower()
    looks_like_zip = is_keras_zip(path)

    # Case A: .keras AND a real zip => load as new Keras format
    if suffix == ".keras" and looks_like_zip:
        log("🧩 Detected valid .keras (zip) file. Loading with Keras 3 loader...")
        try:
            mdl = tf.keras.models.load_model(str(path), compile=False)
            log("✅ Loaded .keras model successfully.")
            return mdl
        except Exception as e:
            log(f"❌ .keras load failed: {type(e).__name__}: {e}")
            raise

    # Case B: .keras BUT NOT a zip => almost certainly HDF5 that was renamed
    if suffix == ".keras" and not looks_like_zip:
        log("🔎 File has .keras extension but is NOT a zip; likely an HDF5 model renamed.")
        # Rename to .h5 next to it (non-destructive copy by rename)
        h5_path = path.with_suffix(".h5")
        try:
            # If a stale .h5 exists, remove it to avoid confusion
            if h5_path.exists():
                log(f"🧹 Removing stale file: {h5_path.name}")
                h5_path.unlink()
            log(f"✏️  Renaming {path.name} -> {h5_path.name} for proper HDF5 loading...")
            path.rename(h5_path)
            path = h5_path
        except Exception as e:
            log(f"⚠️ Rename failed, will try legacy HDF5 loader without rename: {e}")
            # Fall through to legacy HDF5 loader on the original file
            try:
                mdl = _load_h5_legacy(path)
                log("✅ Loaded HDF5 model via legacy loader without rename.")
                return mdl
            except Exception as e2:
                log(f"❌ Legacy HDF5 load (no rename) failed: {type(e2).__name__}: {e2}")
                raise

        # After rename, try normal HDF5 load
        try:
            log("📚 Loading HDF5 model after rename...")
            mdl = tf.keras.models.load_model(str(path), compile=False)
            log("✅ Loaded HDF5 model successfully.")
            return mdl
        except Exception as e:
            log(f"⚠️ Standard HDF5 loader failed after rename: {type(e).__name__}: {e}")
            log("🔁 Trying legacy HDF5 loader path...")
            mdl = _load_h5_legacy(path)
            log("✅ Loaded HDF5 model via legacy loader.")
            return mdl

    # Case C: .h5/.hdf5 extension
    if suffix in {".h5", ".hdf5"}:
        log("🧪 Detected HDF5 model by extension. Loading...")
        try:
            mdl = tf.keras.models.load_model(str(path), compile=False)
            log("✅ Loaded HDF5 model successfully.")
            return mdl
        except Exception as e:
            log(f"⚠️ Standard HDF5 loader failed: {type(e).__name__}: {e}")
            log("🔁 Trying legacy HDF5 loader path...")
            mdl = _load_h5_legacy(path)
            log("✅ Loaded HDF5 model via legacy loader.")
            return mdl

    # Unknown extension — try new Keras first, then legacy HDF5
    log(f"ℹ️ Unknown extension '{suffix}'. Trying .keras loader first...")
    try:
        mdl = tf.keras.models.load_model(str(path), compile=False)
        log("✅ Loaded model with generic loader.")
        return mdl
    except Exception as e:
        log(f"⚠️ Generic loader failed: {type(e).__name__}: {e}")
        log("🔁 Trying legacy HDF5 loader path as a fallback...")
        mdl = _load_h5_legacy(path)
        log("✅ Loaded via legacy HDF5 fallback.")
        return mdl

# -----------------------------
# Image preprocessing
# -----------------------------
def preprocess_image(img: Image.Image) -> np.ndarray:
    # Expect a 28x28 grayscale image for Quick, Draw! models
    img = img.convert("L").resize((28, 28))  # grayscale
    arr = np.array(img).astype("float32") / 255.0
    arr = 1.0 - arr  # invert if training used white-on-black lines
    arr = arr.reshape(1, 28, 28, 1)
    return arr

def load_categories(path: Path) -> list:
    if not path.exists():
        log(f"⚠️ categories file not found at {path.resolve()}, using index labels.")
        return []
    with path.open("r", encoding="utf-8") as f:
        cats = [line.strip() for line in f if line.strip()]
    log(f"📚 Loaded {len(cats)} categories.")
    return cats

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    return jsonify({"ok": True, "message": "QuickDraw AI API"}), 200

@app.get("/health")
def health():
    return jsonify({
        "ok": model is not None,
        "model_loaded": model is not None,
        "categories": len(CATEGORIES)
    }), 200

@app.post("/predict")
def predict():
    if model is None:
        return jsonify({"ok": False, "error": "Model not loaded"}), 503

    # Accept either file upload under 'image' or base64 JSON { "image": "data:image/png;base64,..." }
    if "image" in request.files:
        file = request.files["image"]
        image_bytes = file.read()
    else:
        payload = request.get_json(silent=True) or {}
        data_uri = payload.get("image", "")
        if "," in data_uri:
            image_base64 = data_uri.split(",", 1)[1]
        else:
            image_base64 = data_uri
        try:
            import base64
            image_bytes = base64.b64decode(image_base64)
        except Exception:
            return jsonify({"ok": False, "error": "Invalid image/base64"}), 400

    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        return jsonify({"ok": False, "error": f"Cannot open image: {e}"}), 400

    x = preprocess_image(img)
    preds = model.predict(x, verbose=0)[0].tolist()
    top_idx = int(np.argmax(preds))
    top_score = float(preds[top_idx])

    label = CATEGORIES[top_idx] if CATEGORIES and top_idx < len(CATEGORIES) else str(top_idx)
    return jsonify({
        "ok": True,
        "label": label,
        "index": top_idx,
        "score": top_score,
        "probs": preds
    }), 200

# -----------------------------
# Startup
# -----------------------------
def startup():
    global model, CATEGORIES

    log("🚀 Starting server...")
    try:
        # Helpful environment echoes
        log(f"🧭 ENV PORT={PORT}")
        log(f"🧭 ENV QUICKDRAW_MODEL_PATH={MODEL_PATH_ENV}")
        log(f"🧭 CATEGORIES_FILE={CATEGORIES_PATH}")

        # Load categories
        CATEGORIES = load_categories(CATEGORIES_PATH)

        # Try to load the model
        try:
            model = load_model_robust(MODEL_PATH)
        except Exception as e:
            log(f"❌ ERROR loading model: {type(e).__name__}: {e}")
            log("💡 Hints: If you downloaded an '.h5' file and renamed it to '.keras',")
            log("   please commit it as '.h5' OR let this server rename it on boot as it just did.")
            log("   You can also convert locally and commit a real '.keras' zip via:")
            log("     >>> m = tf.keras.models.load_model('your_model.h5', compile=False)")
            log("     >>> m.save('quickdraw_model.keras')  # this writes a .keras zip")
            model = None

        if model is not None:
            # Show input spec for debugging
            try:
                sig = getattr(model, "inputs", None)
                log(f"🧪 Model inputs: {sig}")
            except Exception:
                pass
            log("✅ Model ready for inference.")
        else:
            log("⚠️ Model not loaded. /predict will return 503 until fixed.")

    except Exception as e:
        log(f"💥 Fatal startup error: {type(e).__name__}: {e}")
        raise

if __name__ == "__main__":
    startup()
    # Run Flask (Render autodetects PORT)
    app.run(host="0.0.0.0", port=PORT, debug=False)
