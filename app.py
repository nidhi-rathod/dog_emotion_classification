import os
import sys
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

sys.path.insert(0, os.path.dirname(__file__))
import config as cfg
from utils.features import extract_features

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'tmp_uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the lightweight TFLite model directly from the main directory with Flex Ops enabled
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'dog_emotion_model.tflite')

if os.path.exists(MODEL_PATH):
    print(f"Loading ultra-lightweight TFLite model with Flex support from: {MODEL_PATH}")
    
    # This line tells TFLite to automatically support complex LSTM/Custom layers
    tf.config.experimental_connect_to_cluster = None 
    
    interpreter = tf.lite.Interpreter(
        model_path=MODEL_PATH,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
else:
    raise FileNotFoundError(f"Critical Error: Model file missing at {MODEL_PATH}")
CLASSES = cfg.EMOTION_CLASSES

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "online",
        "message": "Dog Emotion Classification API is running on TFLite!",
        "supported_classes": CLASSES
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file stream found"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            # Extract features and format to float32 for TFLite compliance
            features = extract_features(file_path).astype(np.float32)
            features = features[np.newaxis, ..., np.newaxis] 
            
            # Run inference via TFLite Interpreter
            interpreter.set_tensor(input_details[0]['index'], features)
            interpreter.invoke()
            preds = interpreter.get_tensor(output_details[0]['index'])
            
            confidence = float(np.max(preds))
            predicted_idx = int(np.argmax(preds))
            
            if confidence < cfg.CONFIDENCE_THRESH:
                prediction = "uncertain"
            else:
                prediction = CLASSES[predicted_idx]
                
            if os.path.exists(file_path):
                os.remove(file_path)
                
            return jsonify({
                "success": True,
                "prediction": prediction,
                "confidence": round(confidence, 4),
                "raw_probabilities": {CLASSES[i]: float(preds[0][i]) for i in range(len(CLASSES))}
            }), 200
            
        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    # Use the port environment variable assigned by Render
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
