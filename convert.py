import tensorflow as tf

# Load your .h5 model
model = tf.keras.models.load_model("quickdraw_model.h5")

# Save to SavedModel format
model.save("quickdraw_saved_model")

print("✅ Model converted! Folder 'quickdraw_saved_model' created.")
