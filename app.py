import os
import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Target the lightweight tabular XGBoost model and encoder in the root folder
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'dog_emotion_xgb.joblib')
ENCODER_PATH = os.path.join(os.path.dirname(__file__), 'label_encoder.joblib')

model = None
label_encoder = None

try:
    print("🔄 Loading lightweight tabular XGBoost engine...")
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
        model = joblib.load(MODEL_PATH)
        label_encoder = joblib.load(ENCODER_PATH)
        print("✅ SUCCESS: Tabular server is live and completely stable!")
    else:
        print(f"❌ ERROR: Model or Encoder files missing from directory. Looking for:\n - {MODEL_PATH}\n - {ENCODER_PATH}")
except Exception as e:
    print(f"❌ CRITICAL LOAD ERROR: {str(e)}")

@app.route("/", methods=["GET"])
def root():
    if model is None or label_encoder is None:
        return jsonify({"status": "offline", "detail": "Model or Encoder failed to load on boot"}), 503
    return jsonify({"status": "online", "detail": "XGBoost audio prediction endpoint is active!"})

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or label_encoder is None:
        return jsonify({"error": "Model offline"}), 503
        
    try:
        data = request.get_json(force=True)
        if not data or 'features' not in data:
            return jsonify({"error": "Missing 'features' data matrix array in payload"}), 400
            
        # Accept the flat list of 140 numbers and shape it into a single 2D row (1, 140)
        features = np.array(data["features"], dtype=np.float32).reshape(1, -1)
        
        # Extract the true bounded probability distribution mapping across all classes
        probabilities = model.predict_proba(features)[0]
        
        # Pull the absolute champion index
        pred_idx = np.argmax(probabilities)
        
        # Extract the clean true confidence probability bound (between 0.0 and 1.0)
        confidence = float(probabilities[pred_idx])
        
        # Decode index back into the clean emotion string ("happy", "aggressive", etc.)
        detected_emotion = label_encoder.inverse_transform([pred_idx])[0]
        
        return jsonify({
            "emotion": str(detected_emotion),
            "confidence": confidence
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"Tabular evaluation failure: {str(e)}"}), 500

if __name__ == "__main__":
    # Dynamic port binding for Render deployment stability
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
