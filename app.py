import os
import numpy as np
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# Target the tabular XGBoost model and the encoder inside your root directory
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'dog_emotion_xgb.joblib')
ENCODER_PATH = os.path.join(os.path.dirname(__file__), 'label_encoder.joblib')

model = None
le = None

try:
    print("🔄 Loading lightweight tabular XGBoost engine...")
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
        model = joblib.load(MODEL_PATH)
        le = joblib.load(ENCODER_PATH)
        print("✅ SUCCESS: Tabular server is live and completely stable!")
    else:
        print(f"❌ ERROR: Model or Encoder files missing from repository. Looking for:\n - {MODEL_PATH}\n - {ENCODER_PATH}")
except Exception as e:
    print(f"❌ CRITICAL LOAD ERROR: {str(e)}")

@app.route("/", methods=["GET"])
def root():
    if model is None or le is None:
        return jsonify({"status": "offline", "detail": "Model or Encoder failed to load on boot"}), 503
    return jsonify({"status": "online", "detail": "XGBoost audio prediction endpoint is active!"})

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or le is None:
        return jsonify({"error": "Model offline"}), 503
        
    try:
        json_data = request.get_json(force=True)
        if not json_data or "features" not in json_data:
            return jsonify({"error": "Missing 'features' data matrix array in payload"}), 400
            
        # Accept the flat list of 140 numbers and shape it into a single 2D row (1, 140)
        features = np.array(json_data["features"], dtype=np.float32).reshape(1, -1)
        
        # Predict class index using XGBoost
        pred_idx = int(model.predict(features)[0])
        
        # Decode index back into the true clean emotion string ("happy", "aggressive", etc.)
        detected_emotion = le.inverse_transform([pred_idx])[0]
        
        # Extract probability matrix score
        probabilities = model.predict_proba(features)[0]
        confidence = float(probabilities[pred_idx])
        
        return jsonify({
            "emotion": detected_emotion,
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"error": f"Tabular evaluation failure: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
