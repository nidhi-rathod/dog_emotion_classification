import librosa
import numpy as np
import config as cfg

def extract_features(file_path_str):
    # Load audio using the sample rate from our config
    y, sr = librosa.load(file_path_str, sr=cfg.SAMPLE_RATE)
    
    # Feature extraction stack
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=cfg.N_MFCC)
    delta = librosa.feature.delta(mfccs)
    delta2 = librosa.feature.delta(mfccs, order=2)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    
    # Spectral contrast — captures tonal vs noise differences (helps angry/fearful separation)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)  # shape (7, T)
    
    # RMS energy contour — prosody feature flagged in both papers
    rms = librosa.feature.rms(y=y)                                      # shape (1, T)
    
    # Merge all features together vertically
    features = np.vstack([mfccs, delta, delta2, chroma, contrast, rms])
    
    # Ensure standard length padding/truncating logic (Shape: 140, 130)
    T = features.shape[1]
    if T > cfg.MAX_PAD_LEN: 
        features = features[:, :cfg.MAX_PAD_LEN]
    else: 
        features = np.pad(features, ((0, 0), (0, cfg.MAX_PAD_LEN - T)), mode='constant')
        
    return features