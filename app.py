import os
import sys
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# Ensure the app can find your local config and utils
sys.path.insert(0, os.path.dirname(__file__))
import config as cfg
from utils.features import extract_features
from utils.losses import focal_loss

app = Flask(__name__)

# Configure a temporary upload folder for cloud server requests
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'tmp_uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the TFLite model instead of the massive H5 file
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'dog_emotion_model.tflite')

if os.path.exists(MODEL_PATH):
    print(f"Loading cloud TFLite model from: {MODEL_PATH}")
    # Initialize TFLite Interpreter
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
else:
    raise FileNotFoundError(f"Critical Error: Model file missing at {MODEL_PATH}. Cannot start server.")

CLASSES = cfg.EMOTION_CLASSES

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "online",
        "message": "Dog Emotion Classification API is running on Render!",
        "supported_classes": CLASSES,
        "confidence_threshold": cfg.CONFIDENCE_THRESH
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Check if an audio file was sent in the request
    if 'file' not in request.files:
        return jsonify({"error": "No file stream found in the request"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected for upload"}), 400

    if file:
        # 2. Save incoming file safely to the cloud server disk temp folder
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            # 3. Extract the exact spectrogram features using your utils pipeline
            features = extract_features(file_path)
            features = features[np.newaxis, ..., np.newaxis] # Reshape to (1, H, W, 1)
            
            # 4. Make a live inference prediction
            preds = model.predict(features, verbose=0)
            confidence = float(np.max(preds))
            predicted_idx = int(np.argmax(preds))
            
            # 5. Apply your team lead's safety confidence rule
            if confidence < cfg.CONFIDENCE_THRESH:
                prediction = "uncertain"
            else:
                prediction = CLASSES[predicted_idx]
                
            # Clean up the audio file from disk after processing to save space
            if os.path.exists(file_path):
                os.remove(file_path)
                
            # 6. Return a clean API JSON package response
            return jsonify({
                "success": True,
                "prediction": prediction,
                "confidence": round(confidence, 4),
                "raw_probabilities": {CLASSES[i]: float(preds[0][i]) for i in range(len(CLASSES))}
            }), 200
            
        except Exception as e:
            # Clean up on failure to keep the cloud environment tidy
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    # Bind to local port 5000 for testing
    app.run(host='0.0.0.0', port=5000, debug=True)
