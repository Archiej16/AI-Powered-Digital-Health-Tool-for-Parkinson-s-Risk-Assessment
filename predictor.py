import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope # <-- Import custom_object_scope
import os
import numpy as np
import joblib 

# Import the preprocessing functions
from .preprocessing import preprocess_image, preprocess_audio

def load_models():
    """
    Loads all final drawing, MRI, and audio models from their saved files.
    """
    models = {}
    
    # --- THIS IS THE FIX ---
    # We define the "unknown" layer as a built-in TensorFlow function.
    custom_objects = {'TrueDivide': tf.math.divide}

    # --- Load Drawing Model ---
    drawing_model_path = os.path.join('models', 'parkinsons_drawings_model_final_4_datasets.h5')
    print(f"Loading drawing model from: {drawing_model_path}")
    if os.path.exists(drawing_model_path):
        # Load the model within the custom_object_scope
        with custom_object_scope(custom_objects):
            models['drawing'] = load_model(drawing_model_path, compile=False)
        print("Drawing model loaded successfully!")
    else:
        print(f"ERROR: Drawing model not found at {drawing_model_path}")
        models['drawing'] = None

    # --- Load MRI Model ---
    mri_model_path = os.path.join('models', 'parkinsons_mri_model_final_combined_regularized.h5')
    print(f"Loading MRI model from: {mri_model_path}")
    if os.path.exists(mri_model_path):
        # Load the model within the custom_object_scope
        with custom_object_scope(custom_objects):
            models['mri'] = load_model(mri_model_path, compile=False)
        print("MRI model loaded successfully!")
    else:
        print(f"ERROR: MRI model not found at {mri_model_path}")
        models['mri'] = None
        
    # --- LOAD FINAL AUDIO MODEL & SCALER ---
    audio_model_path = os.path.join('models', 'parkinsons_audio_model.joblib')
    audio_scaler_path = os.path.join('models', 'audio_scaler.joblib')
    
    print(f"Loading audio model from: {audio_model_path}")
    if os.path.exists(audio_model_path):
        models['audio'] = joblib.load(audio_model_path)
        print("Audio model loaded successfully!")
    else:
        print(f"ERROR: Audio model not found at {audio_model_path}")
        models['audio'] = None
        
    print(f"Loading audio scaler from: {audio_scaler_path}")
    if os.path.exists(audio_scaler_path):
        models['audio_scaler'] = joblib.load(audio_scaler_path)
        print("Audio scaler loaded successfully!")
    else:
        print(f"ERROR: Audio scaler not found at {audio_scaler_path}")
        models['audio_scaler'] = None
        
    return models

def make_drawing_prediction(model, image_bytes: bytes):
    """Makes a prediction on a spiral drawing image."""
    try:
        # 1. Preprocess the image
        image_array = preprocess_image(image_bytes)
        if image_array is None:
            print("Drawing preprocessing failed.")
            return None

        # 2. Get the model's prediction
        prediction_prob = model.predict(image_array)[0][0]

        # 3. Determine the class label and confidence score
        if prediction_prob < 0.5:
            prediction = 'Healthy'
            confidence = (1 - prediction_prob) * 100
        else:
            prediction = 'Parkinson'
            confidence = prediction_prob * 100
            
        return {"prediction": prediction, "confidence": f"{confidence:.2f}"}

    except Exception as e:
        print(f"Error during drawing prediction: {e}")
        return None

def make_mri_prediction(model, image_bytes: bytes):
    """Makes a prediction on an MRI scan image."""
    try:
        # 1. Preprocess the image
        image_array = preprocess_image(image_bytes)
        if image_array is None:
            print("MRI preprocessing failed.")
            return None

        # 2. Get the model's prediction
        prediction_prob = model.predict(image_array)[0][0]

        # 3. Determine the class label and confidence score
        if prediction_prob < 0.5:
            prediction = 'Healthy'
            confidence = (1 - prediction_prob) * 100
        else:
            prediction = 'Parkinson'
            confidence = prediction_prob * 100
            
        return {"prediction": prediction, "confidence": f"{confidence:.2f}"}

    except Exception as e:
        print(f"Error during MRI prediction: {e}")
        return None

def make_audio_prediction(model, scaler, audio_bytes: bytes):
    """Makes a prediction on an audio file."""
    try:
        # 1. Preprocess the audio
        features = preprocess_audio(audio_bytes, scaler)
        if features is None:
            print("Audio preprocessing failed.")
            return None

        # 2. Get the model's prediction
        prediction_class = model.predict(features)[0]
        prediction_probs = model.predict_proba(features)[0]

        # 3. Determine the class label and confidence score
        if prediction_class == 0:
            prediction = 'Healthy'
            confidence = prediction_probs[0] * 100 # Confidence in 'Healthy'
        else:
            prediction = 'Parkinson'
            confidence = prediction_probs[1] * 100 # Confidence in 'Parkinson'
            
        return {"prediction": prediction, "confidence": f"{confidence:.2f}"}

    except Exception as e:
        print(f"Error during audio prediction: {e}")
        return None

