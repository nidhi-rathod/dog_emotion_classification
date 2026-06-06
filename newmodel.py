import os
import numpy as np
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# CONFIGURATION
SAMPLE_RATE = 22050
N_MFCC = 40

DATASET_DIR = r"C:\Users\NIDHI\OneDrive\Documents\dog_emotion_classification\data\raw\dataset" 

def extract_flat_audio_features(file_path):
    """Extracts authentic audio engineering features and flattens them into a 1D row."""
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        if len(y) == 0:
            return None
        
        # 🌟 CRITICAL FIX FOR SHORT AUDIO: 
        # If the audio is shorter than roughly 0.2 seconds (4096 samples), 
        # pad it with trailing zeros (silence) so librosa has enough frames for math operations.
        if len(y) < 4096:
            y = np.pad(y, (0, 4096 - len(y)), mode='constant')
        
        # Extract base MFCCs
        mfccs_matrix = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        
        # 🌟 DOUBLE SAFETY GUARD: Ensure frame width is at least 9 for delta calculations
        if mfccs_matrix.shape[1] < 9:
            mfccs_matrix = np.pad(mfccs_matrix, ((0, 0), (0, 9 - mfccs_matrix.shape[1])), mode='edge')
        
        # Take the mean across time frames (axis=1) to flatten 2D matrices into 1D vectors
        mfccs = np.mean(mfccs_matrix, axis=1)
        delta = np.mean(librosa.feature.delta(mfccs_matrix), axis=1)
        delta2 = np.mean(librosa.feature.delta(mfccs_matrix, order=2), axis=1)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6), axis=1)
        rms = np.mean(librosa.feature.rms(y=y), axis=1)
        
        # Horizontally stack all arrays into a single row of 140 columns
        feature_row = np.hstack([mfccs, delta, delta2, chroma, contrast, rms])
        return feature_row
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# --- AUTOMATICALLY BUILD X_TRAIN AND Y_TRAIN ---
X_train = []
y_train = []

print("🔄 Starting local feature extraction from dataset directory...")

# This expects folders named after emotions (e.g., "Happy", "Aggressive") inside DATASET_DIR
if os.path.exists(DATASET_DIR):
    for emotion_label in os.listdir(DATASET_DIR):
        emotion_folder = os.path.join(DATASET_DIR, emotion_label)
        
        if os.path.isdir(emotion_folder):
            for file_name in os.listdir(emotion_folder):
                if file_name.lower().endswith(('.wav', '.mp3')):
                    file_path = os.path.join(emotion_folder, file_name)
                    
                    features = extract_flat_audio_features(file_path)
                    if features is not None:
                        X_train.append(features)
                        y_train.append(emotion_label)
                        
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    print(f"✅ Extraction complete! Extracted {X_train.shape[0]} samples with {X_train.shape[1]} features.")
else:
    print(f"❌ ERROR: The path '{DATASET_DIR}' does not exist. Please put your real path on Line 13.")
    # Fallback dummy shapes just to clear VS Code syntax compilation checks if path isn't modified yet
    X_train = np.zeros((1, 140))
    y_train = np.array(["Neutral"])

# --- MODEL PROCESSING STACK ---
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Encode string labels to integers (0 to 4) required by XGBoost
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
joblib.dump(le, 'label_encoder.joblib')

# 🌟 CRITICAL SPLIT: Separate data into training (80%) and unseen testing (20%) sets
X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
    X_train, y_train_encoded, test_size=0.2, random_state=42, stratify=y_train_encoded
)

print(f"\n📊 Splitting Dataset: {len(X_train_split)} training samples, {len(X_test_split)} testing validation samples.")
print("-" * 50)

# --- Evaluate Model A: Random Forest ---
print("🌲 Training Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
rf_model.fit(X_train_split, y_train_split)

# Test the Random Forest
rf_predictions = rf_model.predict(X_test_split)
rf_accuracy = accuracy_score(y_test_split, rf_predictions)
joblib.dump(rf_model, 'dog_emotion_rf.joblib')

# --- Evaluate Model B: XGBoost ---
print("🚀 Training XGBoost model...")
xgb_model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train_split, y_train_split)

# Test the XGBoost
xgb_predictions = xgb_model.predict(X_test_split)
xgb_accuracy = accuracy_score(y_test_split, xgb_predictions)
joblib.dump(xgb_model, 'dog_emotion_xgb.joblib')

# --- THE SHOWDOWN: COMPARING ACCURACY MATCHUPS ---
print("\n" + "="*20 + " FINAL ACCURACY SHOWDOWN " + "="*20)
print(f"🌲 Random Forest Accuracy : {rf_accuracy:.2%}")
print(f"🚀 XGBoost Classifier Accuracy: {xgb_accuracy:.2%}")
print("="*65)

if xgb_accuracy > rf_accuracy:
    print("🏆 WINNER: XGBoost is more accurate! Deploy 'dog_emotion_xgb.joblib' to Render.")
else:
    print("🏆 WINNER: Random Forest is more accurate! Deploy 'dog_emotion_rf.joblib' to Render.")

# Print broken-down details for the winning model's precision across emotions
print("\n📝 Detailed Classification Breakdown for the winner:")
winner_preds = xgb_predictions if xgb_accuracy > rf_accuracy else rf_predictions
print(classification_report(y_test_split, winner_preds, target_names=le.classes_))