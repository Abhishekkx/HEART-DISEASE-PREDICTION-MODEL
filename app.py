from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from joblib import load
import warnings
import os
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load Model and Scaler
try:
    model = load_model('heart_disease_model.h5')
    scaler = load('scaler.joblib')
    app.logger.info("Model and scaler loaded successfully")
except FileNotFoundError as e:
    app.logger.error(f"Error loading model or scaler: {e}")
    exit(1)

# Expected Features
expected_features = [
    'age', 'sex', 'trestbps', 'chol', 'fbs', 'restecg', 'thalch', 'exang', 'oldpeak', 
    'slope', 'ca', 'thal', 'sysBP', 'diaBP', 'glucose', 'BPMeds', 'diabetes', 'height', 
    'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active'
]

# Input Validation Ranges
feature_ranges = {
    'age': (0, 120), 'sex': (0, 1), 'trestbps': (0, 300), 'chol': (0, 600), 'fbs': (0, 1),
    'restecg': (0, 3), 'thalch': (0, 220), 'exang': (0, 1), 'oldpeak': (-10, 10), 
    'slope': (0, 3), 'ca': (0, 4), 'thal': (0, 3), 'sysBP': (0, 300), 'diaBP': (0, 200),
    'glucose': (0, 500), 'BPMeds': (0, 1), 'diabetes': (0, 1), 'height': (0, 250),
    'weight': (0, 300), 'ap_hi': (0, 300), 'ap_lo': (0, 200), 'cholesterol': (1, 3),
    'gluc': (1, 3), 'smoke': (0, 1), 'alco': (0, 1), 'active': (0, 1)
}

def validate_input(data):
    for feature, value in data.items():
        if feature in feature_ranges:
            min_val, max_val = feature_ranges[feature]
            if not (min_val <= value <= max_val):
                raise ValueError(f"{feature} value {value} is out of range [{min_val}, {max_val}]")
    return data

def predict(df):
    df = df.reindex(columns=expected_features, fill_value=0)
    df_scaled = scaler.transform(df)
    probs = model.predict(df_scaled, verbose=0).flatten()
    return probs

# Serve the frontend
@app.route('/')
def serve_frontend():
    app.logger.debug("Serving index.html")
    try:
        return send_file('index.html')
    except FileNotFoundError:
        app.logger.error("index.html not found")
        return jsonify({'error': 'Frontend file not found'}), 404

# Single prediction endpoint
@app.route('/api/predict', methods=['POST'])
def predict_single():
    app.logger.debug("Received single prediction request")
    try:
        data = request.json['data']
        df = pd.DataFrame(data)
        for _, row in df.iterrows():
            validate_input(row.to_dict())
        probs = predict(df)
        app.logger.info(f"Prediction successful: {probs.tolist()}")
        return jsonify({'probabilities': probs.tolist()})
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 400

# Batch prediction endpoint
@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    app.logger.debug("Received batch prediction request")
    try:
        file = request.files['file']
        df = pd.read_csv(file)
        for feature in df.columns:
            df[feature] = pd.to_numeric(df[feature], errors='coerce')
            for idx, value in df[feature].items():
                if pd.isna(value):
                    df.at[idx, feature] = 0
                validate_input({feature: df.at[idx, feature]})
        probs = predict(df)
        app.logger.info(f"Batch prediction successful: {len(probs)} predictions")
        return jsonify({'probabilities': probs.tolist()})
    except Exception as e:
        app.logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 400

# Metrics endpoint
@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    app.logger.debug("Received metrics request")
    try:
        # Mock training history (replace with actual data if available)
        metrics = {
            'loss': [0.6, 0.5, 0.4, 0.35, 0.32],
            'val_loss': [0.65, 0.55, 0.45, 0.38, 0.36],
            'accuracy': [0.65, 0.70, 0.72, 0.73, 0.74],
            'val_accuracy': [0.60, 0.68, 0.70, 0.71, 0.74],
            'precision': 0.71,
            'recall': 0.73,
            'f1_score': 0.72
        }
        app.logger.info("Metrics retrieved successfully")
        return jsonify(metrics)
    except Exception as e:
        app.logger.error(f"Metrics error: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.logger.info("Starting Flask server")
    app.run(host='0.0.0.0', port=5000, debug=True)