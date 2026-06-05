import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'dog_emotion_model.h5')
EMOTIONS = ["Aggressive", "Fearful", "Happy", "Neutral", "Pain"]

model = None

try:
    print("🔄 Initializing structural model bypass...")
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
        print("✅ SUCCESS: Model loaded into memory perfectly!")
    else:
        print("❌ ERROR: Weight file not found")
except Exception as e:
    print(f"❌ TF ERROR: {str(e)}")

@app.route("/", methods=["GET"])
def root():
    if model is None:
        return jsonify({"status": "offline", "detail": "Model variable is None"}), 503
    return jsonify({"status": "online", "detail": "Lightweight endpoint is live and stable!"})

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model is offline"}), 503
        
    try:
        # Accept the pre-extracted matrix payload from the local machine
        json_data = request.get_json(force=True)
        if not json_data or "features" not in json_data:
            return jsonify({"error": "Missing 'features' key in payload JSON"}), 400
            
        # Reconstruct the matrix directly into a NumPy array
        data = np.array(json_data["features"], dtype=np.float32)
        
        # Execute the model evaluation safely
        predictions = model.predict(data)
        max_idx = np.argmax(predictions[0])
        
        return jsonify({
            "emotion": EMOTIONS[max_idx],
            "confidence": float(predictions[0][max_idx])
        })
    except Exception as e:
        return jsonify({"error": f"Prediction evaluation failure: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
