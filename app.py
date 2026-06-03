import os
import tempfile
import numpy as np
import librosa
import tensorflow as tf

from fastapi import FastAPI, UploadFile, File, HTTPException

# ==========================================================
# SETTINGS
# ==========================================================

MODEL_PATH = "dog_emotion_model.tflite"

SAMPLE_RATE = 16000
N_MFCC = 40
MAX_PAD_LEN = 130

EMOTIONS = [
    "Angry",
    "Happy",
    "Sad",
    "Fearful",
    "Neutral"
]

# ==========================================================
# FASTAPI
# ==========================================================

app = FastAPI()

interpreter = None
input_details = None
output_details = None

# ==========================================================
# FEATURE EXTRACTION
# ==========================================================

def extract_features_from_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

    mfccs = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC
    )

    delta = librosa.feature.delta(mfccs)

    delta2 = librosa.feature.delta(
        mfccs,
        order=2
    )

    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr
    )

    contrast = librosa.feature.spectral_contrast(
        y=y,
        sr=sr,
        n_bands=6
    )

    rms = librosa.feature.rms(y=y)

    features = np.vstack([
        mfccs,
        delta,
        delta2,
        chroma,
        contrast,
        rms
    ])

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

    global interpreter
    global input_details
    global output_details

    try:
        print("Loading TFLite model...")

        interpreter = tf.lite.Interpreter(
            model_path=MODEL_PATH
        )

        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print("Model loaded successfully")

        print("Input shape:",
              input_details[0]["shape"])

    except Exception as e:
        print("Model loading failed:", str(e))

# ==========================================================
# HEALTH CHECK
# ==========================================================

@app.get("/")
def root():

    if interpreter is None:
        return {
            "status": "initializing"
        }

    return {
        "status": "Dog Emotion TFLite API Engine is running"
    }

# ==========================================================
# PREDICT
# ==========================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if interpreter is None:
        raise HTTPException(
            status_code=503,
            detail="Model still loading"
        )

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as tmp:

            contents = await file.read()
            tmp.write(contents)
            temp_path = tmp.name

        features = extract_features_from_audio(
            temp_path
        )

        os.remove(temp_path)

        # (140,130)
        features = np.expand_dims(
            features,
            axis=-1
        )

        # (140,130,1)

        features = np.expand_dims(
            features,
            axis=0
        )

        # (1,140,130,1)

        expected_shape = tuple(
            input_details[0]["shape"]
        )

        if tuple(features.shape) != expected_shape:

            raise HTTPException(
                status_code=500,
                detail=f"Shape mismatch. "
                       f"Expected {expected_shape}, "
                       f"Got {features.shape}"
            )

        interpreter.set_tensor(
            input_details[0]["index"],
            features
        )

        interpreter.invoke()

        predictions = interpreter.get_tensor(
            output_details[0]["index"]
        )

        predictions = predictions[0]

        emotion_index = int(
            np.argmax(predictions)
        )

        confidence = float(
            predictions[emotion_index]
        )

        return {
            "emotion": EMOTIONS[emotion_index],
            "confidence": confidence
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# ==========================================================
# LOCAL RUN
# ==========================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5000
    )
