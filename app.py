import os
import sys
import json
import numpy as np
import paho.mqtt.client as mqtt

# Tighten TensorFlow memory allocations exactly like your working version
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import tensorflow as tf

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Global TFLite operational states
interpreter = None
input_details = None
output_details = None

# Point directly to your TFLite file
MODEL_PATH = "dog_emotion_model.tflite"

# MQTT Configuration (The hardware team will provide these broker details)
MQTT_BROKER = os.environ.get("MQTT_BROKER", "broker.hivemq.com")  # Using a public test broker by default
MQTT_PORT = int(os.environ.get("MQTT_PORT", 1883))
MQTT_SUB_TOPIC = "dog/+/features"  # '+' is a wildcard, matches any device ID (e.g., dog/collar_01/features)

def load_tflite_model():
    """Initializes and allocates TFLite tensors on script startup."""
    global interpreter, input_details, output_details
    try:
        print("Initializing lightweight TFLite engine into memory...")
        if os.path.exists(MODEL_PATH):
            interpreter = tf.lite.Interpreter(
                model_path=MODEL_PATH,
                experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF
            )
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            print("TFLite Engine online and tensors allocated perfectly!")
            return True
        else:
            print(f"Core load failed: Model file missing at {MODEL_PATH}")
            return False
    except Exception as e:
        print(f"Core load failed: {e}")
        return False

# --- MQTT Callback Handlers ---

def on_connect(client, userdata, flags, rc):
    """Triggered automatically when backend successfully logs into the MQTT broker."""
    if rc == 0:
        print(f"Connected successfully to MQTT Broker: {MQTT_BROKER}")
        # Subscribe to the feature stream topic
        client.subscribe(MQTT_SUB_TOPIC)
        print(f" Listening for device data on topic: {MQTT_SUB_TOPIC}")
    else:
        print(f"Connection failed with result code {rc}")

def on_message(client, userdata, msg):
    """Triggered automatically every time a hardware device publishes audio features."""
    if interpreter is None:
        print("Dropping message: TFLite engine is still initializing.")
        return

    try:
        # Determine which device sent the message from the topic path
        # e.g., "dog/collar_55/features" -> device_id becomes "collar_55"
        topic_parts = msg.topic.split('/')
        device_id = topic_parts[1]
        response_topic = f"dog/{device_id}/predictions"

        # Parse incoming JSON payload
        payload = json.loads(msg.payload.decode('utf-8'))
        features_list = payload.get("features")
        
        if not features_list:
            print(f"Invalid payload received from {device_id}")
            return

        # Convert incoming list into standard float32 array
        input_data = np.array(features_list, dtype=np.float32)
        
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
        
        # Structure the response payload
        result = {
            "device_id": device_id,
            "emotion": emotions[emotion_idx],
            "confidence": round(float(predictions[0][emotion_idx]), 4)
        }
        
        # Publish the response payload right back to the device's unique channel
        client.publish(response_topic, json.dumps(result))
        print(f"Sent prediction to {response_topic}: {emotions[emotion_idx]} ({result['confidence']})")

    except Exception as e:
        print(f"Failed to process payload: {e}")

# --- Main Runtime Loop ---
if __name__ == "__main__":
    # 1. Force the model to load into memory first
    if load_tflite_model():
        # 2. Setup the persistent MQTT Client
        client = mqtt.Client()
        client.on_connect = on_connect
        client.on_message = on_message

        print(f" Connecting to broker {MQTT_BROKER}...")
        client.connect(MQTT_BROKER, MQTT_PORT, 60)

        # 3. Block and listen forever (This keeps your worker process alive)
        client.loop_forever()
