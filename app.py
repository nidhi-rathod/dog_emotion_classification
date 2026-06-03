import os
import sys
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# Ensure the app can find your local config and utils
sys.path.insert(0, os.path.dirname(__file__))
import config as cfg
from utils.features import extract_features  # Using your actual sidebar util file!

app = Flask(__name__)

# Configure a temporary upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'tmp_uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Target your exact model path from the sidebar layout
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'dog_emotion_model.h5')

EMOTIONS = ["Aggressive", "Fearful", "Happy", "Neutral", "Pain"]

# Load your H5 model globally on startup
model = None
try:
    print("🔄 Loading Keras .h5 model from models folder...")
    if os.path.exists(MODEL_PATH):
        # If you use a custom loss like focal_loss, we handle it here
        try:
            from utils.losses import focal_loss
            model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'focal_loss': focal_loss})
        except Exception:
            model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Keras H5 Model loaded successfully!")
    else:
        print(f"❌ ERROR: Model file not found at {MODEL_PATH}")
except Exception as e:
    print(f"❌ CRITICAL: Model initialization failed: {str(e)}")

@app.route("/", methods=["GET"])
def root():
    if model is None:
        return jsonify({"status": "error", "message": "Model failed to load on server"}), 503
    return jsonify({"status": "Dog Emotion Flask API Engine is running smoothly"})

# Change the endpoint to match the /predict route your team is using
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"detail": "Model is offline. Server running but .h5 file failed."}), 503

    # Check if the file wrapper exists in the request
    if 'file' not in request.files:
        return jsonify({"detail": "No file field found in the request payload. Label your key as 'file'."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"detail": "Empty filename transmitted."}), 400

    try:
        # Save file to temporary directory
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Extract features using your custom pipeline script from the sidebar
        features = extract_features(filepath)
        
        # Clean up the file after processing
        if os.path.exists(filepath):
            os.remove(filepath)

        # Shape formatting to match what your specific H5 network structure expects
        # Add batch dimension [1, features...]
        if len(features.shape) == 2:
            features = np.expand_dims(features, axis=0)
        if len(features.shape) == 2: # If it needs a 4D shape like CNN [1, X, Y, 1]
            features = np.expand_dims(features, axis=-1)

        # Run predictions using the standard Keras model syntax
        predictions = model.predict(features)[0]
        
        emotion_index = int(np.argmax(predictions))
        confidence = float(predictions[emotion_index])

        return jsonify({
            "emotion": EMOTIONS[emotion_index],
            "confidence": confidence
        }), 200

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return jsonify({"detail": f"Backend Error: {str(e)}", "trace": error_details}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
