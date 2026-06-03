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
# FASTAPI & MODEL INITIALIZATION (FIXED COLD LOAD)
# ==========================================================

app = FastAPI()

# Load the model globally right away - completely bypassing the broken startup event handler
try:
    print(" Loading TFLite model from disk...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file '{MODEL_PATH}' was not found in your repository directory!")
        
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(" Model loaded successfully into memory!")
    print(" Target Input shape:", input_details[0]["shape"])
except Exception as e:
    print(f" CRITICAL ERROR: Model loading failed: {str(e)}")
    interpreter = None
    input_details = None
    output_details = None

# ==========================================================
# FEATURE EXTRACTION
# ==========================================================

def extract_features_from_audio(audio_path):
    try:
        y, sr = sf.read(audio_path)
    except Exception as e:
        raise ValueError(f"Could not parse WAV file format: {str(e)}")
    
    y = y.astype(np.float32)
    
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)
        
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE

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
# HEALTH CHECK
# ==========================================================

@app.get("/")
def root():
    if interpreter is None:
        return {"status": "error", "message": "Model engine failed to initialize on host server. Check logs."}
    return {"status": "Dog Emotion TFLite API Engine is running smoothly"}

# ==========================================================
# PREDICT
# ==========================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if interpreter is None or input_details is None:
        raise HTTPException(
            status_code=503, 
            detail="Model is offline. The server started up but your .tflite file failed to initialize."
        )

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            contents = await file.read()
            tmp.write(contents)
            temp_path = tmp.name

        features = extract_features_from_audio(temp_path)
        os.remove(temp_path)

        expected_shape = tuple(input_details[0]["shape"])

        if len(expected_shape) == 4:
            if len(features.shape) == 2:
                features = np.expand_dims(features, axis=-1)
            features = np.expand_dims(features, axis=0)
        elif len(expected_shape) == 3:
            if len(features.shape) == 2:
                features = np.expand_dims(features, axis=0)
        else:
            features = np.reshape(features, expected_shape)

        features = features.astype(np.float32)

        interpreter.set_tensor(input_details[0]["index"], features)
        interpreter.invoke()

        raw_output = interpreter.get_tensor(output_details[0]["index"])
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
        raise HTTPException(
            status_code=500, 
            detail=f"Backend Processing Error: {str(e)} | Trace: {error_details}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
