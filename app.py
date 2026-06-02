
import os
import sys
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Tighten TensorFlow memory allocations exactly like your working version
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import tensorflow as tf

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

app = FastAPI()

# Global TFLite operational states
interpreter = None
input_details = None
output_details = None

# Point directly to your TFLite file
MODEL_PATH = "dog_emotion_model.tflite"

@app.on_event("startup")
def load_tflite_model_background():
    global interpreter, input_details, output_details
    try:
        print("🔄 Initializing lightweight TFLite engine into memory...")
        if os.path.exists(MODEL_PATH):
            # Load interpreter with builtin reference operations enabled for custom layer tracking
            interpreter = tf.lite.Interpreter(
                model_path=MODEL_PATH,
                experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF
            )
            interpreter.allocate_tensors()
            
            # Map input/output tensor memory addresses
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            print("✅ TFLite Engine online and tensors allocated perfectly!")
        else:
            print(f"❌ Core load failed: Model file missing at {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Core load failed: {e}")

class AudioFeatures(BaseModel):
    features: list

@app.get("/")
def read_root():
    if interpreter is None:
        return {"status": "Initializing", "message": "Backend online. TFLite engine loading."}
    return {"status": "Dog Emotion TFLite API Engine is running"}

@app.post("/predict")
def predict_emotion(data: AudioFeatures):
    if interpreter is None:
        raise HTTPException(status_code=503, detail="TFLite Engine is still initializing.")
    try:
        # Convert incoming list into standard float32 array
        input_data = np.array(data.features, dtype=np.float32)
        
        # Match tensor dimensions to model expectations
        if input_data.ndim == 3:
            input_data = np.expand_dims(input_data, axis=0)
        elif input_data.ndim == 2:
            input_data = np.expand_dims(input_data, axis=(0, -1))
            
        # Run inference using your allocated TFLite runtime states
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])
        
        emotion_idx = int(np.argmax(predictions[0]))
        emotions = ["Angry", "Happy", "Sad", "Fearful", "Neutral"]
        
        return {
            "emotion": emotions[emotion_idx],
            "confidence": float(predictions[0][emotion_idx])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
