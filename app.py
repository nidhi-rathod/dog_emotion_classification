import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'dog_emotion_model.tflite')
EMOTIONS = ["Aggressive", "Fearful", "Happy", "Neutral", "Pain"]

interpreter = None
input_details = None
output_details = None

try:
    print("🔄 Loading ultra-lightweight TFLite interpreter...")
    if os.path.exists(MODEL_PATH):
        # Initialize TFLite interpreter which uses 90% less RAM than full Keras
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        
        # Track memory block layout shapes
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("✅ SUCCESS: TFLite model allocated in memory cleanly!")
    else:
        print("❌ ERROR: TFLite file not found")
except Exception as e:
    print(f"❌ TFLITE LOADING ERROR: {str(e)}")

@app.route("/", methods=["GET"])
def root():
    if interpreter is None:
        return jsonify({"status": "offline", "detail": "Interpreter variable is None"}), 503
    return jsonify({"status": "online", "detail": "TFLite inference server is live and stable!"})

@app.route("/predict", methods=["POST"])
def predict():
    if interpreter is None:
        return jsonify({"error": "Model is offline"}), 503
        
    try:
        # Accept the pre-extracted matrix payload from the local machine
        json_data = request.get_json(force=True)
        if not json_data or "features" not in json_data:
            return jsonify({"error": "Missing 'features' key in payload JSON"}), 400
            
        # Reconstruct the matrix directly into a NumPy array matching model data types
        data = np.array(json_data["features"], dtype=np.float32)
        
        # Run inference using the allocated TFLite tensor blocks
        interpreter.set_tensor(input_details[0]['index'], data)
        interpreter.invoke()
        
        predictions = interpreter.get_tensor(output_details[0]['index'])
        max_idx = np.argmax(predictions[0])
        
        return jsonify({
            "emotion": EMOTIONS[max_idx],
            "confidence": float(predictions[0][max_idx])
        })
    except Exception as e:
        return jsonify({"error": f"TFLite evaluation failure: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
