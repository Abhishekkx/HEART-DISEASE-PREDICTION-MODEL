import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from joblib import load
import warnings
warnings.filterwarnings('ignore')

# Load Model and Scaler
try:
    model = load_model('heart_disease_model.h5')
    scaler = load('scaler.joblib')
except FileNotFoundError as e:
    print(f"Error loading model or scaler: {e}")
    exit(1)

# Expected Features (from training data)
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

def manual_input():
    print("Enter patient data (0 for missing values where applicable):")
    input_data = {}
    for feature in expected_features:
        while True:
            try:
                value = float(input(f"{feature}: "))
                input_data[feature] = value
                validate_input({feature: value})
                break
            except ValueError as e:
                print(f"Invalid input: {e}. Try again.")
    return pd.DataFrame([input_data])

def load_csv():
    file_path = input("Enter CSV file path: ")
    try:
        df = pd.read_csv(file_path)
        missing_cols = [col for col in expected_features if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing columns {missing_cols}. Filling with 0.")
            for col in missing_cols:
                df[col] = 0
        df = df[expected_features]
        for feature in df.columns:
            df[feature] = pd.to_numeric(df[feature], errors='coerce')
            for idx, value in df[feature].items():
                if pd.isna(value):
                    df.at[idx, feature] = 0
                validate_input({feature: df.at[idx, feature]})
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        exit(1)

def predict(df):
    df = df.reindex(columns=expected_features, fill_value=0)
    df_scaled = scaler.transform(df)
    probs = model.predict(df_scaled, verbose=0).flatten()
    return probs

def main():
    print("Heart Disease Prediction")
    print("1. Manual Input\n2. Load CSV File")
    choice = input("Choose option (1 or 2): ")
    
    if choice == '1':
        df = manual_input()
        probs = predict(df)
        prob = probs[0]
        print(f"\nPredicted risk probability: {prob:.3f}")
        print("\U0001FA7A Prediction:", "\u26A0\uFE0F High Risk" if prob > 0.5 else "\u2705 Low Risk")
    elif choice == '2':
        df = load_csv()
        probs = predict(df)
        for idx, prob in enumerate(probs):
            print(f"\nPatient {idx+1}:")
            print(f"Predicted risk probability: {prob:.3f}")
            print("\U0001FA7A Prediction:", "\u26A0\uFE0F High Risk" if prob > 0.5 else "\u2705 Low Risk")
    else:
        print("Invalid choice. Exiting.")
        exit(1)

if __name__ == "__main__":
    main()