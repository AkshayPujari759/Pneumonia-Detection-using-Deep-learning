# MediScan AI - Pneumonia Diagnostics

MediScan AI is a web application built with Flask and TensorFlow that uses a fine-tuned MobileNetV2 deep learning model to analyze Chest X-Rays and detect the presence of Pneumonia. 

The application features a modern, interactive dashboard with an animated 3D DNA background (built with Three.js) and a seamless upload-to-prediction pipeline.

## Features
- **Real-time Diagnostics**: Upload a chest X-Ray and get an instant prediction ("NORMAL" or "PNEUMONIA") with a confidence score.
- **Deep Learning Powered**: Uses a pre-trained `MobileNetV2` model fine-tuned for X-Ray classification.
- **Modern User Interface**: A dynamic glassmorphism dashboard built with TailwindCSS.
- **Interactive 3D Background**: Features a rotating, theme-responsive 3D DNA strand using Three.js.
- **Dark/Light Mode**: Fully responsive theme toggling built in.

## Tech Stack
- **Backend**: Python, Flask, Flask-CORS
- **Machine Learning**: TensorFlow / Keras, NumPy, Pillow
- **Frontend**: HTML5, Vanilla JavaScript, TailwindCSS, Three.js

## Project Structure
```
Pneumonia_Project/
├── app.py                            # Flask application and model inference logic
├── pneumonia_mobilenetv2_final.h5    # Pre-trained Keras model
├── pneumonia_mobilenet.ipynb         # Jupyter notebook containing model training/fine-tuning
├── requirements.txt                  # Python dependencies
├── templates/
│   └── index.html                    # Main HTML interface
└── static/
    ├── script.js                     # Frontend logic (3D rendering, upload, and prediction)
    └── style.css                     # Custom CSS styles
```

## How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
cd YOUR-REPO-NAME
```

### 2. Install Dependencies
It's recommended to use a virtual environment. Install the required Python packages using:
```bash
pip install -r requirements.txt
```
*(Alternatively, if you face strict versioning issues in the requirements.txt, you can install the core dependencies via: `pip install Flask tensorflow pillow numpy flask-cors`)*

### 3. Run the Application
Start the Flask development server:
```bash
python app.py
```

### 4. Open in your Browser
Navigate to [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in your web browser. Upload an X-ray image and click "Run Diagnosis" to test the model.

## License
This project is open-source and available under the [MIT License](LICENSE).
