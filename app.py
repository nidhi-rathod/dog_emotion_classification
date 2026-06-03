import os
import tensorflow as tf
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'dog_emotion_model.h5')

model = None

try:
    print(f"🔄 Checking path: {MODEL_PATH}")
    if os.path.exists(MODEL_PATH):
        print("🔄 File found! Forcing TensorFlow to load .keras file...")
        # Load without compilation to bypass layer configuration bugs entirely
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("✅ SUCCESS: Model loaded into memory perfectly!")
    else:
        print("❌ ERROR: The file dog_emotion_model.keras does not exist in the root folder!")
except Exception as e:
    print(f"❌ TF ERROR MESSAGE: {str(e)}")

@app.route("/", methods=["GET"])
def root():
    if model is None:
        return jsonify({"status": "offline", "detail": "Model variable is None inside server memory"}), 503
    return jsonify({"status": "online", "detail": "Model is loaded and ready!"})

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model is offline"}), 503
    
    # Simple hardcoded response to test hardware pipeline communication first
    return jsonify({"emotion": "Neutral", "confidence": 0.95})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
