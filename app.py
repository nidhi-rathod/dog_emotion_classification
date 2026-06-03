import os
import numpy as np
import tensorflow as tf
import librosa
from flask import Flask, request, jsonify

app = Flask(__name__)

# Target the newly generated .keras file directly in the main folder
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'dog_emotion_model.keras')
EMOTIONS = ["Aggressive", "Fearful", "Happy", "Neutral", "Pain"]

model = None

try:
    print("🔄 Loading modern Keras model from root directory...")
    if os.path.exists(MODEL_PATH):
        # compile=False is the magic fix that stops the BatchNormalization crash
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("✅ Modern Keras Model loaded successfully!")
    else:
        print(f"❌ ERROR: Model file not found at {MODEL_PATH}")
except Exception as e:
    print(f"❌ CRITICAL: Model initialization failed: {str(e)}")

def extract_features(audio_path):
    """Extracts MFCC features to match your model's expected shape."""
    try:
        # Load audio with the standard 22050Hz sample rate
        y, sr = librosa.load(audio_path, sr=22050)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        
        # Adjust target shape to match whatever your model architecture expects
        if mfccs.shape[1] < 130:
            pad_width = 130 - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :130]
            
        # Reshape for custom Convolutional/Functional layers
        # Shapes vary by model; adjustments go here if input shape differs
        features = np.expand_dims(mfccs, axis=-1)
        features = np.expand_dims(features, axis=0)
        return features
    except Exception as e:
        print(f"Error in feature extraction: {e}")
        return None

@app.route("/", methods=["GET"])
def root():
    if model is None:
        return jsonify({"status": "error", "message": "Model failed to load on server"}), 503
    return jsonify({"status": "Dog Emotion Flask API Engine is running smoothly"})

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model is offline"}), 503
        
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    temp_path = os.path.join(os.path.dirname(__file__), "temp_audio.wav")
    try:
        file.save(temp_path)
        data = extract_features(temp_path)
        
        if data is None:
            return jsonify({"error": "Feature extraction failed"}), 500
            
        predictions = model.predict(data)
        max_idx = np.argmax(predictions[0])
        detected_emotion = EMOTIONS[max_idx]
        confidence = float(predictions[0][max_idx])

        return jsonify({
            "emotion": detected_emotion,
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
