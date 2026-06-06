import streamlit as st
import numpy as np
import requests
import librosa

st.title("Dog Emotion Detection")
st.write("Upload a dog audio clip to detect emotion")

RENDER_BACKEND_URL = "https://dog-emotion-classification.onrender.com/predict"

SAMPLE_RATE = 22050
N_MFCC = 40
MAX_PAD_LEN = 130

def extract_features_locally(uploaded_file):
    """Processes the audio safely using your authentic model formula locally."""
    # Save uploaded data to a temporary local file so librosa can track it
    with open("temp_local_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    y, sr = librosa.load("temp_local_audio.wav", sr=SAMPLE_RATE)
    if len(y) == 0:
        return None
        
    if len(y) < 2048:
        y = np.pad(y, (0, 2048 - len(y)), mode='constant')
        
    fixed_hop = 512
    n_fft = 2048

    # Build your exact 140-dimension feature stack
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    target_t = mfccs.shape[1]

    delta = librosa.feature.delta(mfccs)
    delta2 = librosa.feature.delta(mfccs, order=2)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=fixed_hop, n_fft=n_fft)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6, hop_length=fixed_hop, n_fft=n_fft)
    rms = librosa.feature.rms(y=y, hop_length=fixed_hop, frame_length=n_fft)

    components = [mfccs, delta, delta2, chroma, contrast, rms]
    aligned_components = []
    
    for comp in components:
        current_t = comp.shape[1]
        if current_t > target_t:
            comp = comp[:, :target_t]
        elif current_t < target_t:
            comp = np.pad(comp, ((0, 0), (0, target_t - current_t)), mode='constant')
        aligned_components.append(comp)
        
    features = np.vstack(aligned_components)
    
    T = features.shape[1]
    if T > MAX_PAD_LEN: 
        features = features[:, :MAX_PAD_LEN]
    else: 
        features = np.pad(features, ((0, 0), (0, MAX_PAD_LEN - T)), mode='constant')
        
    features = np.expand_dims(features, axis=-1)
    features = np.expand_dims(features, axis=0)
    return features

uploaded = st.file_uploader("Choose a .wav file", type=["wav"])

if uploaded:
    st.audio(uploaded)
    
    if st.button("Detect Emotion"):
        with st.spinner("Processing audio features locally and analyzing in cloud..."):
            try:
                # 1. Process feature alignment matrix on local system hardware
                local_matrix = extract_features_locally(uploaded)
                if local_matrix is None:
                    st.error("Audio feature extraction failed locally.")
                    st.stop()
                
                # 2. Package array vector cleanly into a standard serializable JSON format
                payload = {"features": local_matrix.tolist()}
                
                # 3. Request inference against stable model container
                response = requests.post(RENDER_BACKEND_URL, json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    emotion = result.get("emotion", "Unknown")
                    confidence = result.get("confidence", 0.0)
                    
                    st.success("Prediction Complete!")
                    st.metric(label="Detected Emotion", value=emotion)
                    st.info(f"Confidence Score: {confidence:.2%}")
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
                    
            except Exception as e:
                st.error(f"Execution Error: {str(e)}")