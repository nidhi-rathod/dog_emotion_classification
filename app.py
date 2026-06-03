import os
import tensorflow as tf

# Target the newly generated .keras file directly in the root folder
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'dog_emotion_model.keras')

EMOTIONS = ["Aggressive", "Fearful", "Happy", "Neutral", "Pain"]

# Global model variable
model = None

try:
    print(" Loading modern Keras .keras model from root directory...")
    if os.path.exists(MODEL_PATH):
        # compile=False bypasses the focal_loss version mismatch errors
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print(" Modern Keras Model loaded successfully!")
    else:
        print(f" ERROR: Model file not found at {MODEL_PATH}")
except Exception as e:
    print(f" CRITICAL: Model initialization failed: {str(e)}")