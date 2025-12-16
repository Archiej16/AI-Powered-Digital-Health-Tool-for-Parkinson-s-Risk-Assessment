import os
import io
import sys
import joblib
import numpy as np
import tensorflow as tf
import librosa
from pathlib import Path
from PIL import Image

# Web Server Libraries
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# MRI Processing Libraries
try:
    import nibabel as nib
    import scipy.ndimage
except ImportError:
    nib = None
    scipy = None
    print("WARNING: 'nibabel' or 'scipy' not found. MRI .nii processing will not work.")

# --- Configuration ---
IMG_SIZE = 224
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static" / "public"

# UPDATED: Looking for .keras files
MODEL_FILES = {
    "audio_model": "audio_model.joblib",
    "audio_scaler": "audio_scaler.joblib",
    "drawing_model": "drawings_model.keras", # Changed from .h5
    "mri_model": "mri_model.keras"           # Changed from .h5
}

def get_model_path(filename):
    path = MODEL_DIR / filename
    return path if path.exists() else None

# --- 1. Load Models ---
print("\n--- Initializing Parkinson's Diagnostic Server ---")
models = {}

try:
    # Load Audio
    a_path = get_model_path(MODEL_FILES["audio_model"])
    s_path = get_model_path(MODEL_FILES["audio_scaler"])
    if a_path and s_path:
        print(f"Loading Audio: {a_path.name}")
        models['audio'] = joblib.load(a_path)
        models['scaler'] = joblib.load(s_path)
    else:
        print(f"❌ WARNING: Audio models missing in {MODEL_DIR}")

    # Load Drawing (.keras)
    d_path = get_model_path(MODEL_FILES["drawing_model"])
    if d_path:
        print(f"Loading Drawing: {d_path.name}")
        # .keras files load natively without errors
        models['drawing'] = tf.keras.models.load_model(str(d_path))
    else:
        print(f"❌ WARNING: Drawing model missing ({MODEL_FILES['drawing_model']})")

    # Load MRI (.keras)
    m_path = get_model_path(MODEL_FILES["mri_model"])
    if m_path:
        print(f"Loading MRI: {m_path.name}")
        models['mri'] = tf.keras.models.load_model(str(m_path))
    else:
        print(f"❌ WARNING: MRI model missing ({MODEL_FILES['mri_model']})")

    print("--- Models Loaded ---\n")

except Exception as e:
    print(f"CRITICAL ERROR loading models: {e}")

# --- 2. Preprocessing Functions (Unchanged) ---

def preprocess_audio_features(audio_bytes):
    try:
        y, sr = librosa.load(io.BytesIO(audio_bytes), duration=30, sr=None)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)
        spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)
        spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr).T, axis=0)
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr).T, axis=0)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)
        features = np.concatenate((mfccs, chroma, mel, contrast, tonnetz, spec_cent, spec_bw, rolloff, zcr))
        return features.reshape(1, -1)
    except: return None

def preprocess_drawing(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = image.resize((IMG_SIZE, IMG_SIZE))
        return np.expand_dims(np.array(image), axis=0)
    except: return None

def preprocess_mri(file_path_or_bytes, filename):
    try:
        if filename.lower().endswith(('.nii', '.nii.gz')):
            if not nib: return None
            temp_path = "temp_mri.nii.gz"
            with open(temp_path, "wb") as f: f.write(file_path_or_bytes)
            img = nib.load(temp_path)
            data = img.get_fdata()
            if os.path.exists(temp_path): os.remove(temp_path)
            if data.max() > 0: data = data / data.max() * 255.0
            slice_idx = data.shape[-1] // 2
            img_slice = data[..., slice_idx]
            img_slice_resized = scipy.ndimage.zoom(img_slice, (IMG_SIZE / img_slice.shape[0], IMG_SIZE / img_slice.shape[1]), order=1)
            image_array = np.stack((img_slice_resized,) * 3, axis=-1)
            return np.expand_dims(image_array, axis=0)
        else:
            image = Image.open(io.BytesIO(file_path_or_bytes)).convert('RGB')
            image = image.resize((IMG_SIZE, IMG_SIZE))
            return np.expand_dims(np.array(image), axis=0)
    except: return None

# --- 3. FastAPI App Setup ---
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
async def read_index():
    if (STATIC_DIR / 'index.html').exists():
        return FileResponse(STATIC_DIR / 'index.html')
    return {"error": "Frontend not found"}

@app.post("/predict/audio")
async def predict_audio(file: UploadFile = File(...)):
    if 'audio' not in models: raise HTTPException(500, "Audio model failed to load.")
    features = preprocess_audio_features(await file.read())
    if features is None: raise HTTPException(400, "Invalid audio")
    probs = models['audio'].predict_proba(models['scaler'].transform(features))[0]
    return {"prediction": "Parkinson's Detected" if probs[1] > 0.5 else "Healthy", "confidence": float((probs[1] if probs[1] > 0.5 else probs[0]) * 100)}

@app.post("/predict/drawing")
async def predict_drawing(file: UploadFile = File(...)):
    if 'drawing' not in models: raise HTTPException(500, "Drawing model failed to load.")
    img = preprocess_drawing(await file.read())
    if img is None: raise HTTPException(400, "Invalid image")
    try:
        prob = models['drawing'].predict(img)[0][0]
        return {"prediction": "Parkinson's Detected" if prob > 0.5 else "Healthy", "confidence": float((prob if prob > 0.5 else 1-prob) * 100)}
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")

@app.post("/predict/mri")
async def predict_mri(file: UploadFile = File(...)):
    if 'mri' not in models: raise HTTPException(500, "MRI model failed to load.")
    img = preprocess_mri(await file.read(), file.filename)
    if img is None: raise HTTPException(400, "Invalid MRI")
    try:
        prob = models['mri'].predict(img)[0][0]
        return {"prediction": "Parkinson's Detected" if prob > 0.5 else "Healthy", "confidence": float((prob if prob > 0.5 else 1-prob) * 100)}
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)