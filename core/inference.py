import os
import json
import tempfile

import numpy as np
import tensorflow as tf

from core.vision import is_valid_leaf

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING (GLOBAL - LOAD ONCE)
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "trained_model.h5")
TREATMENTS_PATH = os.path.join(BASE_DIR, "data", "treatments.json")

model = None
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# CLASS NAMES
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# ─────────────────────────────────────────────────────────────────────────────
# SUSTAINABLE TREATMENTS (loaded from data/treatments.json)
# ─────────────────────────────────────────────────────────────────────────────
with open(TREATMENTS_PATH, "r", encoding="utf-8") as f:
    SUSTAINABLE_TREATMENTS = json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def predict_disease(image):
    """
    Predicts plant disease from image using TensorFlow preprocessing.
    CRITICAL: Uses the EXACT same preprocessing pipeline as training (0-255 pixel values, NO normalization)
    Returns: (plant_name, disease_name, confidence_str, treatments_json, error_message)
    """
    if model is None:
        return None, None, None, None, "❌ Model not loaded. Check server logs."

    if image is None:
        return None, None, None, None, "❌ No image uploaded."

    try:
        # Convert PIL Image to numpy array for validation
        img_array = np.array(image)

        # Validate leaf
        is_valid, validation_message = is_valid_leaf(img_array)
        if not is_valid:
            return None, None, None, None, f"❌ {validation_message}"

        # ✅ CRITICAL FIX: Use TensorFlow's preprocessing to match training pipeline
        # Save PIL image to temporary file for TensorFlow compatibility
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            image.save(tmp_file.name)
            tmp_path = tmp_file.name

        try:
            # Load image using TensorFlow's preprocessing (same as training)
            img = tf.keras.preprocessing.image.load_img(
                tmp_path, target_size=(128, 128)
            )

            # Convert to array using TensorFlow's method (same as training)
            img_array = tf.keras.preprocessing.image.img_to_array(img)

            # ✅ NO NORMALIZATION - Model was trained on 0-255 pixel values
            # Add batch dimension
            img_batch = np.expand_dims(img_array, axis=0)

            # Predict
            predictions = model.predict(img_batch, verbose=0)
            predicted_class_idx = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][predicted_class_idx]) * 100

            predicted_class = CLASS_NAMES[predicted_class_idx]
            parts = predicted_class.split("___")
            plant_name = parts[0].replace("_", " ") if len(parts) > 0 else "Unknown"
            disease_name = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"

            treatments = SUSTAINABLE_TREATMENTS.get(
                predicted_class,
                {
                    "biological": ["No specific treatment available"],
                    "organic": ["No specific treatment available"],
                    "cultural": ["No specific treatment available"],
                },
            )

            confidence_str = f"{confidence:.2f}%"

            return plant_name, disease_name, confidence_str, treatments, None

        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except Exception as e:
        return None, None, None, None, f"❌ Error: {str(e)}"
