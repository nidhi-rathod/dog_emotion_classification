import os
import sys
import json
import threading
import numpy as np
import paho.mqtt.client as mqtt
from fastapi import FastAPI

# Tighten TensorFlow memory allocations exactly like your working version
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import tensorflow as tf

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# --- DUMMY WEB SERVER TO EXPLOIT THE FREE TIER ---
app = FastAPI()

@app.get("/")
def free_tier_ping():
    return {"status": "MQTT Backend is tricking Render into keeping us alive for free!"}

# --- GLOBAL TFLITE STATES ---
interpreter = None
input_details = None
output_details = None
MODEL_PATH = "dog_emotion_model.tflite"

# MQTT Configuration
MQTT_BROKER = os.environ.get("MQTT_BROKER", "broker.hivemq.com")
MQTT_PORT = int(os.environ.get("MQTT_PORT", 1883))
MQTT_SUB_TOPIC = "dog/+/features"

def load_tflite_model():
    global interpreter, input_details, output_details
    try:
        print("🔄 Initializing lightweight TFLite engine into memory...")
        if os.path.exists(MODEL_PATH):
            interpreter = tf.lite.Interpreter(
                model_path=MODEL_PATH,
                experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF
            )
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

# --- MQTT CALLBACKS ---
def on_connect(client, userdata, flags, rc, *args):
    # Handles both old paho-mqtt integer return codes and new version objects
    connection_code = rc if isinstance(rc, int) else rc.value
    if connection_code == 0:
        print(f"🌐 Connected successfully to MQTT Broker: {MQTT_BROKER}")
        client.subscribe(MQTT_SUB_TOPIC)
        print(f"📥 Listening for device data on topic: {MQTT_SUB_TOPIC}")
    else:
        print(f"❌ Connection failed with result code {connection_code}")

def on_message(client, userdata, msg):
    if interpreter is None:
        return
    try:
        topic_parts = msg.topic.split('/')
        device_id = topic_parts[1]
        response_topic = f"dog/{device_id}/predictions"

        payload = json.loads(msg.payload.decode('utf-8'))
        features_list = payload.get("features")
        
        if not features_list:
            return

        input_data = np.array(features_list, dtype=np.float32)
        if input_data.ndim == 3:
            input_data = np.expand_dims(input_data, axis=0)
        elif input_data.ndim == 2:
            input_data = np.expand_dims(input_data, axis=(0, -1))
            
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])
        
        emotion_idx = int(np.argmax(predictions[0]))
        emotions = ["Angry", "Happy", "Sad", "Fearful", "Neutral"]
        
        result = {
            "device_id": device_id,
            "emotion": emotions[emotion_idx],
            "confidence": round(float(predictions[0][emotion_idx]), 4)
        }
        
        client.publish(response_topic, json.dumps(result))
        print(f"🚀 Sent prediction to {response_topic}: {emotions[emotion_idx]}")
    except Exception as e:
        print(f"❌ MQTT processing error: {e}")

def run_mqtt_loop():
    """Runs the MQTT listener inside an isolated parallel execution thread."""
    if load_tflite_model():
        # Version-safe configuration for paho-mqtt (supports v1.x and v2.x)
        try:
            client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
        except AttributeError:
            client = mqtt.Client() # Fallback for older library versions
            
        client.on_connect = on_connect
        client.on_message = on_message
        
        print(f"🔄 Connecting to broker {MQTT_BROKER}...")
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_forever()

# Automatically spin up the MQTT engine in the background when the file loads
mqtt_thread = threading.Thread(target=run_mqtt_loop, daemon=True)
mqtt_thread.start()
