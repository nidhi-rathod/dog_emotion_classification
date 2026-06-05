import os
import numpy as np
import tensorflow as tf
import scipy.signal as signal
from flask import Flask, request, jsonify

app = Flask(__name__)

# CONFIGURATION CONFIG PIPELINE
SAMPLE_RATE = 22050
N_MFCC = 40
MAX_PAD_LEN = 130  # Matrix shape sequence width target (140, 130)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'dog_emotion_model.h5')
EMOTIONS = ["Aggressive", "Fearful", "Happy", "Neutral", "Pain"]

model = None

try:
    print("🔄 Initializing structural bypass...")
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
        print("✅ SUCCESS: Model loaded into memory perfectly!")
    else:
        print("❌ ERROR: File not found")
except Exception as e:
    print(f"❌ TF ERROR: {str(e)}")

def extract_features_pure_math(file_path_str):
    """Authentic mathematical replication of the audio engineering stack using pure SciPy/NumPy."""
    try:
        # Read the wav file using scipy to bypass librosa/soundfile backends completely
        from scipy.io import wavfile
        sr, y = wavfile.read(file_path_str)
        
        # Convert stereo to mono if needed
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
            
        # Normalize to 16-bit float range
        y = y.astype(np.float32) / 32768.0
        
        # Resample safely if hardware output rate doesn't match 22050
        if sr != SAMPLE_RATE:
            num_samples = int(round(len(y) * SAMPLE_RATE / sr))
            y = signal.resample(y, num_samples)
            sr = SAMPLE_RATE

        if len(y) == 0:
            return None

        # Pad with zeros if file length is shorter than standard window frame configurations
        if len(y) < 2048:
            y = np.pad(y, (0, 2048 - len(y)), mode='constant')

        n_fft = 2048
        hop_length = 512

        # 1. Generate clean Short-Time Fourier Transform Spectrogram via SciPy
        frequencies, times, stft_matrix = signal.stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length, nfft=n_fft)
        spectrogram = np.abs(stft_matrix)
        target_t = spectrogram.shape[1]

        # 2. Replicate MFCCs via direct Log-Mel Spectrogram dot product equations
        # This builds a stable filterbank without requiring librosa optimization components
        mel_bins = np.linspace(0, sr / 2, N_MFCC + 2)
        hz_to_mel = lambda hz: 2595 * np.log10(1 + hz / 700.0)
        mel_to_hz = lambda mel: 700 * (10**(mel / 2595.0) - 1)
        mel_points = np.linspace(hz_to_mel(0), hz_to_mel(sr / 2), N_MFCC + 2)
        hz_points = mel_to_hz(mel_points)
        bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
        
        fb = np.zeros((N_MFCC, int(n_fft / 2 + 1)))
        for m in range(1, N_MFCC + 1):
            for k in range(bin_points[m - 1], bin_points[m]):
                fb[m - 1, k] = (k - bin_points[m - 1]) / (bin_points[m] - bin_points[m - 1])
            for k in range(bin_points[m], bin_points[m + 1]):
                fb[m - 1, k] = (bin_points[m + 1] - k) / (bin_points[m + 1] - bin_points[m])
                
        mel_spectrogram = np.dot(fb, spectrogram)
        log_mel_spec = np.log(np.maximum(mel_spectrogram, 1e-5))
        
        # Compute pure Discrete Cosine Transform for MFCCs
        from scipy.fftpack import dct
        mfccs = dct(log_mel_spec, type=2, axis=0, norm='ortho')[:N_MFCC, :]

        # 3 & 4. Standard numerical gradients for Delta and Delta2 arrays
        delta = np.gradient(mfccs, axis=1) if mfccs.shape[1] >= 3 else np.zeros_like(mfccs)
        delta2 = np.gradient(delta, axis=1) if delta.shape[1] >= 3 else np.zeros_like(mfccs)

        # 5. Chroma Pitch Representation via simple octave wrapping math
        chroma = np.zeros((12, target_t))
        for i, freq in enumerate(frequencies):
            if freq > 0:
                pitch_note = int(round(12 * np.log2(freq / 440.0))) % 12
                chroma[pitch_note, :] += spectrogram[i, :]
        chroma_norm = np.max(chroma, axis=0)
        chroma = np.divide(chroma, chroma_norm, out=np.zeros_like(chroma), where=chroma_norm != 0)

        # 6. Spectral Contrast approximation (splits spectrogram matrix into frequency bands)
        contrast = np.zeros((7, target_t))
        band_limits = [0, 200, 500, 1000, 2000, 5000, 10000, sr // 2]
        for b in range(6):
            idx = np.where((frequencies >= band_limits[b]) & (frequencies < band_limits[b + 1]))[0]
            if len(idx) > 0:
                band_spec = spectrogram[idx, :]
                peaks = np.percentile(band_spec, 95, axis=0)
                valleys = np.percentile(band_spec, 5, axis=0)
                contrast[b, :] = np.log(np.maximum(peaks, 1e-5)) - np.log(np.maximum(valleys, 1e-5))
        contrast[6, :] = np.mean(contrast[:6, :], axis=0)

        # 7. RMS Energy tracking logic
        hop_samples = 512
        num_frames = int(np.floor((len(y) - n_fft) / hop_samples)) + 1
        rms = np.zeros((1, num_frames))
        for t in range(num_frames):
            start = t * hop_samples
            rms[0, t] = np.sqrt(np.mean(y[start:start + n_fft]**2))

        # Explicit component vertical structural matching configuration sequence
        components = [mfccs, delta, delta2, chroma, contrast, rms]
        aligned_components = []
        for comp in components:
            current_t = comp.shape[1]
            if current_t > target_t:
                comp = comp[:, :target_t]
            elif current_t < target_t:
                comp = np.pad(comp, ((0, 0), (0, target_t - current_t)), mode='constant')
            aligned_components.append(comp)

        # Merge vertically to 140 features
        features = np.vstack(aligned_components)

        # Truncating / Padding target layout rules (Shape: 140, 130)
        T = features.shape[1]
        if T > MAX_PAD_LEN:
            features = features[:, :MAX_PAD_LEN]
        else:
            features = np.pad(features, ((0, 0), (0, MAX_PAD_LEN - T)), mode='constant')

        features = np.expand_dims(features, axis=-1)
        features = np.expand_dims(features, axis=0)
        return features
    except Exception as e:
        print(f"Pure math structural processing error: {str(e)}")
        return None

@app.route("/", methods=["GET"])
def root():
    if model is None:
        return jsonify({"status": "offline", "detail": "Model variable is None inside server memory"}), 503
    return jsonify({"status": "online", "detail": "Model running completely on stable SciPy pipeline!"})

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
        # Save incoming audio file locally inside cloud storage
        file.save(temp_path)
        
        # Execute stable matrix feature calculations
        data = extract_features_pure_math(temp_path)
        if data is None:
            return jsonify({"error": "Feature tracking layout calculation mismatch processing error"}), 400
            
        # Run inference against raw weights
        predictions = model.predict(data)
        max_idx = np.argmax(predictions[0])
        
        return jsonify({
            "emotion": EMOTIONS[max_idx],
            "confidence": float(predictions[0][max_idx])
        })
    except Exception as e:
        return jsonify({"error": f"Internal matrix evaluation runtime crash: {str(e)}"}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
