import os
import tempfile
import numpy as np
import librosa
import tensorflow as tf
import soundfile as sf

from fastapi import FastAPI, UploadFile, File, HTTPException

# ==========================================================
# SETTINGS
# ==========================================================

MODEL_PATH = "dog_emotion_model.tflite"

SAMPLE_RATE = 22050
N_MFCC = 40
MAX_PAD_LEN = 130

EMOTIONS = [
    "Aggressive",
    "Fearful",
    "Happy",
    "Neutral",
    "Pain"
]

# ==========================================================
# FASTAPI
# ==========================================================

app = FastAPI()

interpreter = None
input_details = None
output_details = None

# ==========================================================
# FEATURE EXTRACTION (FREE TIER PURE-PYTHON BYPASS)
# ==========================================================

def extract_features_from_audio(audio_path):
    try:
        # soundfile can read almost any type of .wav format perfectly!
        y, sr = sf.read(audio_path)
    except Exception as e:
        raise ValueError(f"Could not parse WAV file format: {str(e)}")
    
    # Ensure data is converted to float32 arrays for librosa processing
    y = y.astype(np.float32)
    
    # If the audio file is stereo (2 channels), downmix it to mono (1 channel)
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)
        
    # Normalize volume amplitude to a -1.0 to 1.0 range
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    # Resample on the fly if the uploaded file isn't exactly 22050Hz
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE

    # Extract features using librosa's mathematical processors
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfccs)
    delta2 = librosa.feature.delta(mfccs, order=2)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
    rms = librosa.feature.rms(y=y)

    features = np.vstack([mfccs, delta, delta2, chroma, contrast, rms])
    T = features.shape[1]

    if T > MAX_PAD_LEN:
        features = features[:, :MAX_PAD_LEN]
    else:
        features = np.pad(
            features,
            ((0, 0), (0, MAX_PAD_LEN - T)),
            mode="constant"
        )

    return features.astype(np.float32)

# ==========================================================
# STARTUP
# ==========================================================

@app.on_event("startup")
def load_model():
    global interpreter, input_details, output_details
    try:
        print("Loading TFLite model...")
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("Model loaded successfully")
        print("Input shape:", input_details[0]["shape"])
    except Exception as e:
        print("Model loading failed:", str(e))

# ==========================================================
# HEALTH CHECK
# ==========================================================

@app.get("/")
def root():
    if interpreter is None:
        return {"status": "initializing"}
    return {"status": "Dog Emotion TFLite API Engine is running"}

# ==========================================================
# PREDICT
# ==========================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if interpreter is None:
        raise HTTPException(status_code=503, detail="Model still loading")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            contents = await file.read()
            tmp.write(contents)
            temp_path = tmp.name

        features = extract_features_from_audio(temp_path)
        os.remove(temp_path)

        # 1. READ TARGET MODEL SHAPE DYNAMICALLY
        expected_shape = tuple(input_details[0]["shape"])
        print(f"📊 Audio Features Shape: {features.shape} | Expected Shape: {expected_shape}")

        # 2. MATCH INDICES ACCORDING TO MODEL EXPECTATIONS DYNAMICALLY
        if len(expected_shape) == 4:
            # For 4D convolutional shapes like [1, X, Y, 1]
            if len(features.shape) == 2:
                features = np.expand_dims(features, axis=-1)
            features = np.expand_dims(features, axis=0)
        elif len(expected_shape) == 3:
            # For 3D recurrent/dense shapes like [1, X, Y]
            if len(features.shape) == 2:
                features = np.expand_dims(features, axis=0)
        else:
            # Universal fallback layout match
            features = np.reshape(features, expected_shape)

        features = features.astype(np.float32)

        # 3. SET DATA PACKET INTO TFLITE MEMORY SLOT
        interpreter.set_tensor(input_details[0]["index"], features)
        interpreter.invoke()

        # 4. SAFELY CATCH OUTPUT ARRAY WITH FALLBACK CHECK
        raw_output = interpreter.get_tensor(output_details[0]["index"])
        
        if raw_output is None:
            raise ValueError("TensorFlow Engine returned NoneType instead of predictions. Check model compatibility.")

        predictions = raw_output[0]
        emotion_index = int(np.argmax(predictions))
        confidence = float(predictions[emotion_index])

        return {
            "emotion": EMOTIONS[emotion_index],
            "confidence": confidence
        }

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print("====== BACKEND CRASH LOG ======")
        print(error_details)
        print("===============================")
        raise HTTPException(
            status_code=500, 
            detail=f"Backend Error: {str(e)} | Context Trace: {error_details}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
