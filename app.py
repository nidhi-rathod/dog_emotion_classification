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
    print(" Initializing structural bypass...")
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
        print(" SUCCESS: Model loaded into memory!")
    else:
        print(" ERROR: File not found")
except Exception as e:
    print(f" TF ERROR: {str(e)}")

def extract_features_authentic(file_path_str):
    """Your authentic extraction stack built defensively to prevent shape conflicts."""
    # Load audio safely
    y, sr = librosa.load(file_path_str, sr=SAMPLE_RATE)
    
    if len(y) == 0:
        return None
        
    # Ensure minimum audio length required for spectral feature tracking window frames
    if len(y) < 2048:
        y = np.pad(y, (0, 2048 - len(y)), mode='constant')
    
    # Base MFCC extraction step
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    
    # Force identical frame length alignment (T) across all calculated arrays
    fixed_hop = 512
    n_fft = 2048
    
    # Calculate audio variants matching the precise timeline length of the base MFCC array
    delta = librosa.feature.delta(mfccs)
    delta2 = librosa.feature.delta(mfccs, order=2)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=fixed_hop, n_fft=n_fft)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6, hop_length=fixed_hop, n_fft=n_fft)  
    rms = librosa.feature.rms(y=y, hop_length=fixed_hop, frame_length=n_fft)                                       
    
    # Explicitly truncate/pad each individual component to match MFCC frame count exactly before vertical stacking
    target_t = mfccs.shape[1]
    
    components = [mfccs, delta, delta2, chroma, contrast, rms]
    aligned_components = []
    
    for comp in components:
        current_t = comp.shape[1]
        if current_t > target_t:
            comp = comp[:, :target_t]
        elif current_t < target_t:
            comp = np.pad(comp, ((0, 0), (0, target_t - current_t)), mode='constant')
        aligned_components.append(comp)
        
    # Stack all features together vertically (Guaranteed identical horizontal row sizing)
    features = np.vstack(aligned_components)
    
    # Final matrix sequence length processing adjustments (Shape target: 140, 130)
    T = features.shape[1]
    if T > MAX_PAD_LEN: 
        features = features[:, :MAX_PAD_LEN]
    else: 
        features = np.pad(features, ((0, 0), (0, MAX_PAD_LEN - T)), mode='constant')
        
    # Inject batch and structural channel dimension footprints
    features = np.expand_dims(features, axis=-1)
    features = np.expand_dims(features, axis=0)
    return features

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
        # Save incoming payload sound file locally inside cloud storage
        file.save(temp_path)
        
        # Execute aligned feature tracking array extraction 
        data = extract_features_authentic(temp_path)
        if data is None:
            return jsonify({"error": "Feature array extraction failed due to unreadable audio format"}), 400
            
        # Calculate real inference values directly against your neural network layers
        predictions = model.predict(data)
        max_idx = np.argmax(predictions[0])
        detected_emotion = EMOTIONS[max_idx]
        confidence = float(predictions[0][max_idx])
        
        return jsonify({
            "emotion": detected_emotion,
            "confidence": confidence
        })
    except Exception as e:
        # Return precise code traceback description strings directly over the API payload
        return jsonify({"error": f"Internal processing pipeline failure exception: {str(e)}"}), 500
    finally:
        # Clean local cache footprints cleanly
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
