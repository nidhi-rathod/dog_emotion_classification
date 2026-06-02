import os
from pathlib import Path

# =========================
# AUDIO SETTINGS
# =========================
SAMPLE_RATE       = 22050
CLIP_DURATION     = 3
TOP_DB            = 30
N_MFCC            = 40
MAX_PAD_LEN       = 130

# =========================
# TRAINING SETTINGS
# =========================
BATCH_SIZE        = 16
EPOCHS            = 80
CONFIDENCE_THRESH = 0.45

# =========================
# LOCAL DATA DIRECTORIES 
# =========================
DATA_RAW_DIR   = "./data/raw/dataset"
DATA_PROC_DIR  = "./data/processed"
FEATURES_DIR   = "./data/features"
UNLABELED_DIR  = "./data/unlabeled"
NEW_SOUNDS_DIR = "./data/new_sounds"
OUTPUT_DIR     = "./outputs"
MODEL_DIR      = "./models"

# Create required folders automatically on your computer
os.makedirs(DATA_PROC_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(UNLABELED_DIR, exist_ok=True)
os.makedirs(NEW_SOUNDS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# =========================
# EMOTION CLASSES
# =========================
EMOTION_CLASSES = [
    "aggressive",
    "fearful",
    "happy",
    "neutral",
    "pain"
]

# =========================
# BREED CLASSES
# =========================
BREED_CLASSES = [
    "Beagle",
    "Border Collie",
    "Dachshund",
    "German Shepherd",
    "Golden Retriever",
    "Labrador Retriever",
    "Pomeranian",
    "Pug",
    "Shih Tzu",
    "Siberian Husky"
]

# =========================
# MODEL PATHS
# =========================
KERAS_PATH  = os.path.join(MODEL_DIR, "dog_emotion_model.keras")
TFLITE_PATH = os.path.join(MODEL_DIR, "dog_emotion_model.tflite")

BREED_MODEL_PATH = os.path.join(MODEL_DIR, "dog_breed_model.keras")
BREED_TFLITE_PATH = os.path.join(MODEL_DIR, "dog_breed_model.tflite")

# =========================
# VERIFICATION LOGIC (Runs automatically when you test this file)
# =========================
if __name__ == "__main__":
    print("config.py loaded successfully")
    print("\nVerifying emotion dataset folders locally:\n")
    all_ok = True

    for cls in EMOTION_CLASSES:
        p = Path(DATA_RAW_DIR) / cls
        if p.exists():
            n = len(list(p.glob("*.wav"))) + len(list(p.glob("*.mp3")))
            status = "✓" if n > 0 else "✗ EMPTY"
        else:
            n = 0
            status = "✗ MISSING"

        print(f"{status:10} {cls:<15} {n} audio files")
        if not (p.exists() and n > 0):
            all_ok = False

    if all_ok:
        print("\n✓ All emotion folders found locally!")
    else:
        print("\n⚠ Stop! Some emotion folders are missing or empty in your 'data/raw/dataset/' folder.")