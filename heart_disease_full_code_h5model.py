# 📌 Heart Disease Prediction - Full Combined Code

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump, load

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout

# 2. Load Datasets
uci = pd.read_csv("heart_disease_uci.csv")
fram = pd.read_csv("framingham.csv")
cardio = pd.read_csv("cardio_train.csv", sep=';')

# 3. Clean and Standardize Datasets
uci['target'] = uci['num'].apply(lambda x: 1 if x > 0 else 0)
uci['sex'] = uci['sex'].map({'Male': 1, 'Female': 0})
uci['fbs'] = uci['fbs'].astype(str).str.upper().map({'TRUE': 1, 'FALSE': 0})
uci['exang'] = uci['exang'].astype(str).str.upper().map({'TRUE': 1, 'FALSE': 0})
for col in ['cp', 'restecg', 'slope', 'thal']:
    uci[col] = LabelEncoder().fit_transform(uci[col].astype(str))
uci.drop(['id', 'dataset', 'num'], axis=1, errors='ignore', inplace=True)

fram.rename(columns={'TenYearCHD': 'target', 'male': 'sex'}, inplace=True)
fram.fillna(fram.median(numeric_only=True), inplace=True)

cardio.rename(columns={'cardio': 'target', 'gender': 'sex'}, inplace=True)
cardio['sex'] = cardio['sex'].map({1: 1, 2: 0})
cardio['age'] = (cardio['age'] / 365).astype(int)
cardio.drop(columns=['id'], errors='ignore', inplace=True)

# 4. Align Columns for Merge
uci_cols = ['age', 'sex', 'trestbps', 'chol', 'fbs', 'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
fram_cols = ['age', 'sex', 'sysBP', 'diaBP', 'glucose', 'BPMeds', 'diabetes', 'target']
cardio_cols = ['age', 'sex', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'target']

uci = uci[[col for col in uci_cols if col in uci.columns]]
fram = fram[[col for col in fram_cols if col in fram.columns]]
cardio = cardio[[col for col in cardio_cols if col in cardio.columns]]

# 5. Merge All Datasets
df = pd.concat([uci, fram, cardio], axis=0, ignore_index=True)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(df.median(numeric_only=True), inplace=True)

# 6. Split Features and Target
X = df.drop('target', axis=1)
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
dump(scaler, 'scaler.joblib')

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# 7. Define and Train Neural Network
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=30, batch_size=64, validation_split=0.1)

# 8. Evaluate Model
y_pred = model.predict(X_test).flatten()
y_pred_class = (y_pred > 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred_class))
print("Classification Report:\n", classification_report(y_test, y_pred_class))
sns.heatmap(confusion_matrix(y_test, y_pred_class), annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 8b. Plot Training Curves
plt.figure(figsize=(12, 4))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# 9. Save Model
model.save('heart_disease_model.h5')

# 10. Predict User Input
def predict_user_input():
    model = load_model('heart_disease_model.h5')
    scaler = load('scaler.joblib')

    input_data = {
        'age': 21, 'sex': 1, 'trestbps': 120, 'chol': 180, 'restecg': 1,
        'thalch': 150, 'oldpeak': 1.0, 'slope': 1, 'ca': 0, 'thal': 1,
        'sysBP': 120, 'diaBP': 80, 'glucose': 85, 'BPMeds': 0, 'diabetes': 0,
        'height': 170, 'weight': 75, 'ap_hi': 120, 'ap_lo': 80,
        'cholesterol': 1, 'gluc': 1, 'smoke': 0, 'alco': 0, 'active': 1
    }

    df_input = pd.DataFrame([input_data])
    df_input = df_input.reindex(columns=X.columns, fill_value=0)
    df_scaled = scaler.transform(df_input)
    prob = model.predict(df_scaled)[0][0]

    print(f"Predicted risk probability: {prob:.3f}")
    print("\U0001FA7A Prediction:", "\u26A0\uFE0F High Risk" if prob > 0.5 else "\u2705 Low Risk")

# Run Prediction
df.columns = df.columns.astype(str)  # Ensure column names are strings
predict_user_input()
