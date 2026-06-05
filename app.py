import os
import numpy as np
import tensorflow as tf
import librosa
from flask import Flask, request, jsonify

app = Flask(__name__)

# CONFIGURATION VALUES FROM YOUR LOCAL PIPELINE CONFIG
SAMPLE_RATE = 22050
N_MFCC = 40
MAX_PAD_LEN = 130  # Matches your model's expected shape sequence length (140, 130)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'dog_emotion_model.h5')
EMOTIONS = ["Aggressive", "Fearful", "Happy", "Neutral", "Pain"]

model = None

try:
    print("🔄 Initializing structural bypass...")
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
        print("✅ SUCCESS: Model loaded into memory!")
    else:
        print("❌ ERROR: File not found")
except Exception as e:
    print(f"❌ TF ERROR: {str(e)}")

def extract_features_authentic(file_path_str):
    """Your authentic extraction stack made completely crash-proof for Linux architecture."""
    try:
        # Force the audioread decoding backend to prevent missing libsndfile system crashes
        y, sr = librosa.load(file_path_str, sr=SAMPLE_RATE, res_type='kaiser_fast')
        
        # If the audio is completely empty, prevent a crash
        if len(y) == 0:
            raise ValueError("Audio file is completely empty or has 0 samples.")
            
        # If the audio is too short for spectral_contrast (under 2048 samples), pad it
        if len(y) < 2048:
            y = np.pad(y, (0, 2048 - len(y)), mode='constant')
        
        # Feature extraction stack
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        delta = librosa.feature.delta(mfccs)
        delta2 = librosa.feature.delta(mfccs, order=2)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # Spectral contrast — shape (7, T)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)  
        
        # RMS energy contour — shape (1, T)
        rms = librosa.feature.rms(y=y)                                       
        
        # Merge all features together vertically
        features = np.vstack([mfccs, delta, delta2, chroma, contrast, rms])
        
        # Ensure standard length padding/truncating logic (Shape: 140, 130)
        T = features.shape[1]
        if T > MAX_PAD_LEN: 
            features = features[:, :MAX_PAD_LEN]
        else: 
            features = np.pad(features, ((0, 0), (0, MAX_PAD_LEN - T)), mode='constant')
            
        # Add batch and channel dimensions for CNN/Functional input layers
        features = np.expand_dims(features, axis=-1)
        features = np.expand_dims(features, axis=0)
        return features, None
    except Exception as e:
        # Pass the exact crash string back up the pipeline to expose the issue
        return None, str(e)

@app.route("/", methods=["GET"])
def root():
    if model is None:
        return jsonify({"status": "offline", "detail": "Model variable is None inside server memory"}), 503
    return jsonify({"status": "online", "detail": "Model is loaded and ready with custom feature pipeline!"})

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
        # Save the incoming hardware/Streamlit audio file temporarily
        file.save(temp_path)
        
        # Process the audio file through your authentic feature extractor
        data, pipeline_error = extract_features_authentic(temp_path)
        if data is None:
            # Displays the real mathematical/system exception description on your screen
            return jsonify({
                "error": "Feature array extraction failed", 
                "diagnostic_details": pipeline_error
            }), 400
            
        # Generate the real prediction matrix from the loaded weights
        predictions = model.predict(data)
        max_idx = np.argmax(predictions[0])
        detected_emotion = EMOTIONS[max_idx]
        confidence = float(predictions[0][max_idx])
        
        return jsonify({
            "emotion": detected_emotion,
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"error": f"Prediction runtime crash: {str(e)}"}), 500
    finally:
        # Clean up the temporary file right away to preserve disk space
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
