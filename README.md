Multimodal Parkinson's Disease Detection System

A comprehensive, AI-powered diagnostic tool designed to detect early signs of Parkinson's disease using multiple non-invasive biomarkers: Voice, Visual Motor (Drawings), and Neuroimaging (MRI).

📌 Project Overview
Parkinson's disease is a complex neurodegenerative disorder that manifests in various ways. Relying on a single symptom for diagnosis can be unreliable. This project bridges the gap by integrating three distinct diagnostic modalities into a single, unified platform:
1. Audio Analysis: Detects vocal impairment (e.g., tremors, hypophonia).
2. Visual Motor Analysis: Analyzes hand-drawn spirals and waves to detect motor control issues (e.g., bradykinesia).
3. Neuroimaging Analysis: Examines structural MRI brain scans for neurological changes.
The system is deployed as a user-friendly web application, allowing for real-time analysis and prediction.

🚀 Key Features
- End-to-End Solution: From raw data ingestion to a deployed web application.
- Multimodal AI: Combines Machine Learning (Random Forest) and Deep Learning (CNNs) for robust predictions.
- Advanced Preprocessing: Custom pipelines handle complex medical data formats like 3D NIfTI (.nii) files and raw audio waveforms.
- Modern Tech Stack: Built with a high-performance FastAPI backend and a responsive HTML/Tailwind CSS frontend.
- Real-Time Inference: Asynchronous server architecture ensures fast feedback for users.

🛠️ Tech Stack & Methodology
1. Artificial Intelligence Models
Modality  Algorithm Architecture Accuracy  Key Libraries
Audio  Random Forest  Ensemble Learning  ~80%  librosa, scikit-learn
Drawing DenseNet201  Transfer Learning(CNN) ~84%. TensorFlow, Keras
MRI  Xception  Transfer Learning(CNN)  ~96%  TensorFlow, nibabel

Audio: Extracts 58 acoustic features (MFCCs, Chroma, Spectral Contrast) from .wav files.
Drawings: Uses Computer Vision to analyze line irregularity and pen pressure patterns in spiral drawings.
MRI: Implements a custom pipeline to slice 3D brain scans into 2D cross-sections and normalize them for deep learning analysis.

2. Backend Server
Framework: FastAPI (Python)
Features:
- Asynchronous request handling (async/await).
- RESTful API endpoints for each modality (/predict/audio, /predict/drawing, /predict/mri).
- CORS middleware for secure frontend communication.
- Custom file handling for temporary storage and processing of uploads.

3. Frontend Interface
Technologies: HTML5, JavaScript (ES6+), Tailwind CSS.
Features:
- Dark-mode UI for visual comfort.
- Drag-and-drop file upload zones.
- Dynamic result visualization with confidence scores.

📂 Project StructureParkinsons Codes/
├── backend/
│   ├── app.py                 # Main FastAPI server script
│   ├── models/                # Directory for trained model files
│   │   ├── audio_model.joblib
│   │   ├── audio_scaler.joblib
│   │   ├── drawings_model.keras
│   │   └── mri_model.keras
│   ├── notebooks/             # Jupyter notebooks for model training
│   │   ├── audio_training.ipynb
│   │   ├── multi_data_training.ipynb
│   │   └── final_mri_training.ipynb
│   └── setup_models.py        # Helper script to manage model files
├── static/
│   └── public/
│       └── index.html         # Frontend interface
├── .gitignore
└── README.md

⚙️ Installation & SetupPrerequisites
Python 3.8+
Pip (Python Package Manager)

Step 1: Clone the Repository
git clone <repository-url>
cd "Parkinsons Codes"

Step 2: Install Dependencies
pip install fastapi uvicorn python-multipart tensorflow scikit-learn joblib librosa nibabel scipy pillow aiofiles

Step 3: Prepare the Models
If you have just cloned the repo, you may need to train the models first.
1. Run the Jupyter notebooks in backend/notebooks/ to train the models.
2. Run the setup script to move the trained models to the correct backend folder:
python backend/setup_models.py

Step 4: Run the Application
Start the FastAPI server:
python backend/app.py

Step 5: Access the DashboardOpen your browser and navigate to:http://127.0.0.1:8000/

🧪 Usage Guide
Select a Test: Choose between Voice Analysis, Drawing Analysis, or MRI Scan.
Upload Data:
Audio: Upload a .wav recording of a prolonged vowel sound (e.g., "aaaa").
Drawing: Upload a .png or .jpg image of a hand-drawn spiral.
MRI: Upload a .nii (NIfTI) file or a standard image slice.
Get Results: The system will process the file and display a "Healthy" or "Parkinson's Detected" result along with a confidence percentage.

🚧 Future ImprovementsMultimodal Fusion: Implementing a "meta-learner" to combine predictions from all three models into a single, highly accurate diagnostic score.3D MRI Analysis: Extending the MRI model to use 3D Convolutional Networks (3D-CNNs) for volumetric analysis instead of slice-based classification.Mobile Application: developing a React Native app to allow users to perform the voice and drawing tests directly on their smartphones.
