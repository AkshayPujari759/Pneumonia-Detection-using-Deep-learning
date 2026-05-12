import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allows the frontend to talk to the backend easily

# --- LOAD MODEL ---
# Get the path to the current folder
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'pneumonia_mobilenetv2_final.h5')

print(f"Loading model from: {MODEL_PATH}")
model = None
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully")
except Exception as e:
    print(f"Warning: Could not load model: {e}")
    print("  Predictions will fail until model is placed at:", MODEL_PATH)

def predict_image(image_file):
    # Check if model is loaded
    if model is None:
        raise RuntimeError(f"Model not available. Please place 'pneumonia_mobilenetv2_final.h5' in {PROJECT_ROOT}")
    
    # 1. Load and Preprocess
    image = Image.open(image_file).convert("RGB")
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    
    # 2. Convert to array and normalize
    img_array = np.asarray(image)
    normalized_image_array = (img_array.astype(np.float32) / 255.0)
    data = np.expand_dims(normalized_image_array, axis=0)
    
    # 3. Predict
    prediction = model.predict(data)
    sigmoid_value = float(prediction[0][0])
    
    # 4. Determine Result
    if sigmoid_value > 0.5:
        return "PNEUMONIA", sigmoid_value * 100
    else:
        return "NORMAL", (1 - sigmoid_value) * 100

# --- ROUTES ---

@app.route('/', methods=['GET'])
def home():
    # This serves your cool new HTML page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check for file
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    f = request.files['file']
    if f.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Run prediction logic
        label, confidence = predict_image(f)
        
        # IMPORTANT: Return JSON, not render_template
        return jsonify({
            "prediction": label, 
            "confidence": round(confidence, 2)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)