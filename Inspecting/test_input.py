# File: src/test_input.py

import joblib
import pandas as pd

# --- Step 1: Load preprocessing pipeline and best model ---
pipeline = joblib.load("outputs/preprocessing_pipeline.pkl")
model = joblib.load("outputs/best_model.pkl")

# --- Step 2: Create a test input sample (structured + text columns) ---

# NOTE: Make sure the columns match the expected format from training
sample_input = pd.DataFrame([{
    # Structured features
    "Sex": "Female",
    "Pain": "Yes",
    "Injury": "No",
    "Mental": "Alert",
    "Group": "Local ED",
    "Arrival mode": "Walking",
    "Disposition": "Discharge",
    "Age": 45,
    "Patients number per hour": 10,
    "NRS_pain": 6,
    "SBP": 120,
    "DBP": 80,
    "HR": 85,
    "RR": 18,
    "BT": 36.5,
    "Saturation": 98,
    "Length of stay_min": 45,
    "KTAS duration_min": 10,

    # Text features
    "Chief_complain": "chest pain and dizziness",
    "Diagnosis in ED": "possible myocardial infarction"
}])

# --- Step 3: Apply preprocessing ---
X_processed = pipeline.transform(sample_input)

# --- Step 4: Predict ---
prediction = model.predict(X_processed)
print(f"Predicted KTAS Level: {prediction[0]}")
