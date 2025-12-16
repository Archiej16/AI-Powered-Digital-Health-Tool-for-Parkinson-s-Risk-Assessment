import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import joblib 

# We must import the libraries for processing .nii and .wav files
try:
    import nibabel as nib
    import scipy.ndimage
    import librosa
except ImportError:
    # This won't stop the server but will print a warning.
    # The server will only crash if a .nii or .wav file is uploaded.
    print("WARNING: 'nibabel', 'scipy', or 'librosa' not installed. File processing may fail.")
    print("Please ensure your virtual environment is active and run:")
    print("pip install -r requirements.txt")
    nib = None
    scipy = None
    librosa = None


IMG_SIZE = 224 # We used 224x224 for our final models

# --- Image Preprocessing Functions ---

def _load_nifti_image(image_bytes: bytes):
    """Loads a .nii file from in-memory bytes, extracts a 2D slice, and formats it."""
    # .nii files must be read from disk, so we save to a temp file
    temp_file_path = "temp_nii_file.nii.gz"
    try:
        # Write the in-memory bytes to a temporary file
        with open(temp_file_path, 'wb') as tmp_f:
            tmp_f.write(image_bytes)
        
        # Load the .nii file from the temp file
        img = nib.load(temp_file_path)
        data = img.get_fdata()
        
        # Clean up the temp file immediately
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        # Normalize data to 0-255 range
        if data.max() > 0:
            data = data / data.max()
        data = data * 255.0
        
        # Get the middle slice from the 3rd dimension (axial slice)
        slice_idx = data.shape[-1] // 2
        img_slice = data[..., slice_idx]
        
        # Resize the 2D slice to our model's expected input size
        img_slice_resized = scipy.ndimage.zoom(img_slice, (IMG_SIZE / img_slice.shape[0], IMG_SIZE / img_slice.shape[1]), order=1)
        
        # Convert the 2D grayscale slice to a 3-channel (RGB) image
        image_array = np.stack((img_slice_resized,) * 3, axis=-1)
        
        return image_array.astype(np.float32)

    except Exception as e:
        print(f"Error processing NIfTI file: {e}")
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return None


def _load_standard_image(image_bytes: bytes):
    """Loads a .png, .jpg, or .jpeg file from in-memory bytes."""
    try:
        # Open the image directly from the in-memory bytes
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        # Resize to our model's expected input size
        image = image.resize((IMG_SIZE, IMG_SIZE))
        image_array = np.array(image, dtype=np.float32)
        return image_array
    except Exception as e:
        print(f"Error processing standard image: {e}")
        return None

def preprocess_image(image_bytes: bytes):
    """
    Detects file type and preprocesses image bytes into a model-ready NumPy array.
    """
    image_array = None
    
    # Check for .nii file signature (gzip magic bytes)
    if image_bytes.startswith(b'\x1f\x8b'):
        print("NIfTI file detected. Processing as 3D scan.")
        if nib:
            image_array = _load_nifti_image(image_bytes)
        else:
            print("ERROR: nibabel library not loaded. Cannot process .nii file.")
            return None
    else:
        print("Standard image file detected. Processing as 2D drawing.")
        image_array = _load_standard_image(image_bytes)

    if image_array is None:
        return None

    # We do NOT apply xception.preprocess_input here.
    # It was baked into the model we trained and saved, so it will be
    # applied automatically when we call model.predict().
    
    # Just expand the dimensions to create a "batch" of one image
    image_array_expanded = np.expand_dims(image_array, axis=0)
    
    return image_array_expanded

# --- Audio Preprocessing Functions ---

def _extract_audio_features(audio_bytes: bytes):
    """Extracts features from audio bytes using librosa."""
    if not librosa:
        print("ERROR: librosa library not loaded. Cannot process audio file.")
        return None
        
    temp_file_path = "temp_audio_file.wav"
    try:
        # Librosa needs a file path. Save bytes to a temp file.
        with open(temp_file_path, 'wb') as tmp_f:
            tmp_f.write(audio_bytes)

        # Load the audio file from the temp path
        y, sr = librosa.load(temp_file_path, duration=30) 
        
        # Clean up the temp file immediately
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        # Extract all features as used in the notebook
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)
        spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)
        spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr).T, axis=0)
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr).T, axis=0)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)
        
        # Concatenate all features into a single vector
        features = np.concatenate((mfccs, chroma, mel, contrast, tonnetz, spec_cent, spec_bw, rolloff, zcr))
        return features
        
    except Exception as e:
        print(f"Error processing audio file: {e}")
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return None

def preprocess_audio(audio_bytes: bytes, scaler: joblib.load):
    """
    Preprocesses audio bytes into a scaled feature vector for the ML model.
    """
    features = _extract_audio_features(audio_bytes)
    if features is None:
        return None
    
    # Reshape features to (1, n_features) for the scaler
    features_reshaped = features.reshape(1, -1)
    
    # Scale the features using the pre-fitted scaler
    try:
        features_scaled = scaler.transform(features_reshaped)
    except Exception as e:
        print(f"Error applying scaler: {e}. Check if scaler was loaded.")
        return None
        
    return features_scaled

