import os
import sys

# Set Numba configurations at the absolute initialization boundary
os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ["NUMBA_CACHE_DIR"] = "/tmp/numba_cache"

import numpy as np
import tensorflow as tf
import librosa
from flask import Flask, request, jsonify

app = Flask(__name__)

# CONFIGURATION LAYOUT BOUNDARIES
SAMPLE_RATE = 22050
N_MFCC = 40
MAX_PAD_LEN = 130  # Matrix shape sequence width target (140, 130)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'dog_emotion_model.h5')
EMOTIONS = ["Aggressive", "Fearful", "Happy", "Neutral", "Pain"]

model = None

try:
    print("🔄 Initializing structural model bypass...")
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
        print("✅ SUCCESS: Model loaded into memory perfectly!")
    else:
        print("❌ ERROR: File not found")
except Exception as e:
    print(f"❌ TF ERROR: {str(e)}")

def extract_features_authentic(file_path_str):
    """Extraction pipeline with complete architectural array safety guards."""
    try:
        # Load audio safely
        y, sr = librosa.load(file_path_str, sr=SAMPLE_RATE)
        if len(y) == 0:
            return None
            
        if len(y) < 2048:
            y = np.pad(y, (0, 2048 - len(y)), mode='constant')
            
        fixed_hop = 512
        n_fft = 2048

        # 1. Base MFCC extraction step
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        target_t = mfccs.shape[1]

        # Safe feature generation blocks with zero-fill fallbacks to guarantee 140 dimensions
        try:
            delta = librosa.feature.delta(mfccs)
        except Exception:
            delta = np.zeros_like(mfccs)

        try:
            delta2 = librosa.feature.delta(mfccs, order=2)
        except Exception:
            delta2 = np.zeros_like(mfccs)

        try:
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=fixed_hop, n_fft=n_fft)
        except Exception:
            chroma = np.zeros((12, target_t))

        try:
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6, hop_length=fixed_hop, n_fft=n_fft)
        except Exception:
            contrast = np.zeros((7, target_t))

        try:
            rms = librosa.feature.rms(y=y, hop_length=fixed_hop, frame_length=n_fft)
        except Exception:
            rms = np.zeros((1, target_t))

        components = [mfccs, delta, delta2, chroma, contrast, rms]
        aligned_components = []
        
        for comp in components:
            current_t = comp.shape[1]
            if current_t > target_t:
                comp = comp[:, :target_t]
            elif current_t < target_t:
                comp = np.pad(comp, ((0, 0), (0, target_t - current_t)), mode='constant')
            aligned_components.append(comp)
            
        # Merge all arrays vertically
        features = np.vstack(aligned_components)
        
        # Ensure target shape metrics (140, 130)
        T = features.shape[1]
        if T > MAX_PAD_LEN: 
            features = features[:, :MAX_PAD_LEN]
        else: 
            features = np.pad(features, ((0, 0), (0, MAX_PAD_LEN - T)), mode='constant')
            
        features = np.expand_dims(features, axis=-1)
        features = np.expand_dims(features, axis=0)
        return features
    except Exception as e:
        print(f"Extraction critical failure: {str(e)}")
        return None

@app.route("/", methods=["GET"])
def root():
    if model is None:
        return jsonify({"status": "offline", "detail": "Model variable is None inside server memory"}), 503
    return jsonify({"status": "online", "detail": "Model running completely on stable isolated pipeline!"})

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
        data = extract_features_authentic(temp_path)
        
        if data is None:
            return jsonify({"error": "Feature tracking calculation mismatch processing error"}), 400
            
        predictions = model.predict(data)
        max_idx = np.argmax(predictions[0])
        
        return jsonify({
            "emotion": EMOTIONS[max_idx],
            "confidence": float(predictions[0][max_idx])
        })
    except Exception as e:
        return jsonify({"error": f"Internal matrix runtime crash: {str(e)}"}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
