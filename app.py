import os
import sys
import json
import tempfile
import threading
import numpy as np
import librosa
import paho.mqtt.client as mqtt
from scipy.io import wavfile
from fastapi import FastAPI

# Tighten TensorFlow memory allocations exactly like your working version
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import tensorflow as tf

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# --- DUMMY WEB SERVER TO KEEP RENDER FREE TIER ALIVE ---
app = FastAPI()

@app.get("/")
def free_tier_ping():
    return {"status": "MQTT Audio Engine is tricking Render into keeping us alive for free!"}

# ==========================================================
# SETTINGS & CONFIGURATION
# ==========================================================
MODEL_PATH = "dog_emotion_model.tflite"
SAMPLE_RATE = 22050
N_MFCC = 40
MAX_PAD_LEN = 130

EMOTIONS = ["Aggressive", "Fearful", "Happy", "Neutral", "Pain"]

# Global TFLite Operational States
interpreter = None
input_details = None
output_details = None

# MQTT Settings (Will default to public testing server)
MQTT_BROKER = os.environ.get("MQTT_BROKER", "broker.hivemq.com")
MQTT_PORT = int(os.environ.get("MQTT_PORT", 1883))
MQTT_SUB_TOPIC = "dog/+/audio"  # Matches any device ID (e.g., dog/collar_01/audio)

# ==========================================================
# CORE FEATURE EXTRACTION ENGINE
# ==========================================================
def extract_features_from_audio(audio_path):
    # Pure Python WAV reader - completely bypasses the need for libsndfile1 OS package!
    sr, y = wavfile.read(audio_path)
    
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
        features = np.pad(features, ((0, 0), (0, MAX_PAD_LEN - T)), mode="constant")

    return features.astype(np.float32)

def load_tflite_model():
    global interpreter, input_details, output_details
    try:
        print("🔄 Initializing lightweight TFLite engine into memory...")
        if os.path.exists(MODEL_PATH):
            interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            print("✅ TFLite Engine online and tensors allocated perfectly!")
            return True
        print(f"❌ Core load failed: Model file missing at {MODEL_PATH}")
        return False
    except Exception as e:
        print(f"❌ Core load failed: {e}")
        return False

# ==========================================================
# MQTT BACKGROUND NETWORK HANDLERS
# ==========================================================
def on_connect(client, userdata, flags, rc, *args):
    connection_code = rc if isinstance(rc, int) else rc.value
    if connection_code == 0:
        print(f"🌐 Connected successfully to MQTT Broker: {MQTT_BROKER}")
        client.subscribe(MQTT_SUB_TOPIC)
        print(f"📥 Listening for device audio binary on topic: {MQTT_SUB_TOPIC}")
    else:
        print(f"❌ Connection failed with result code {connection_code}")

def on_message(client, userdata, msg):
    if interpreter is None:
        return
    try:
        # Extract the device ID from the topic (e.g., "dog/collar_99/audio" -> "collar_99")
        topic_parts = msg.topic.split('/')
        device_id = topic_parts[1]
        response_topic = f"dog/{device_id}/predictions"

        print(f"📥 Received raw audio stream payload from device: {device_id}")

        # The device publishes the raw binary content of a .wav file directly as the payload!
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(msg.payload)
            temp_path = tmp.name

        # Process audio and delete temp file
        features = extract_features_from_audio(temp_path)
        os.remove(temp_path)

        features = np.expand_dims(features, axis=-1)
        features = np.expand_dims(features, axis=0)

        # Run model inference
        interpreter.set_tensor(input_details[0]["index"], features)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]["index"])[0]

        emotion_index = int(np.argmax(predictions))
        confidence = float(predictions[emotion_index])

        # Pack prediction payload
        result = {
            "device_id": device_id,
            "emotion": EMOTIONS[emotion_index],
            "confidence": round(confidence, 4)
        }

        # Send back to the specific device's listening channel
        client.publish(response_topic, json.dumps(result))
        print(f"🚀 Sent prediction to {response_topic}: {EMOTIONS[emotion_index]} ({result['confidence']})")

    except Exception as e:
        print(f"❌ MQTT Processing Error: {e}")

def run_mqtt_loop():
    if load_tflite_model():
        try:
            client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
        except AttributeError:
            client = mqtt.Client()
            
        client.on_connect = on_connect
        client.on_message = on_message
        
        print(f"🔄 Connecting to broker {MQTT_BROKER}...")
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_forever()

# Automatically spin up the MQTT loop on a clean separate CPU thread
mqtt_thread = threading.Thread(target=run_mqtt_loop, daemon=True)
mqtt_thread.start()
