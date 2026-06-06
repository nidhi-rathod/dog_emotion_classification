import os
import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the lightweight models cleanly from the root directory
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'dog_emotion_xgb.joblib')
ENCODER_PATH = os.path.join(os.path.dirname(__file__), 'label_encoder.joblib')

model = None
label_encoder = None

try:
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
        model = joblib.load(MODEL_PATH)
        label_encoder = joblib.load(ENCODER_PATH)
        print("🏆 Winner XGBoost model and encoder loaded successfully!")
    else:
        print("❌ ERROR: Joblib files are missing from the directory.")
except Exception as e:
    print(f"💥 Critical Initialization Error: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or label_encoder is None:
        return jsonify({"error": "Model is not initialized on the server"}), 500

    try:
        data = request.get_json()
        if not data or 'features' Bone not in data:
            return jsonify({"error": "Missing 'features' key in JSON payload"}), 400

        # Convert incoming JSON list back to a 1D or 2D NumPy array row
        features = np.array(data['features'], dtype=np.float32).reshape(1, -1)

        # 🌟 CRITICAL FIX FOR THE ACCURACY PROBABILITY PERCENTAGE:
        # Use predict_proba to get clean bounded distributions between 0.0 and 1.0!
        probabilities = model.predict_proba(features)[0]
        pred_idx = np.argmax(probabilities)
        confidence = float(probabilities[pred_idx])

        # Get the human-readable emotion string back from the encoder mapping
        emotion_string = label_encoder.inverse_transform([pred_idx])[0]

        return jsonify({
            "emotion": str(emotion_string),
            "confidence": confidence
        }), 200

    except Exception as e:
        return jsonify({"error": f"Internal prediction failure: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
