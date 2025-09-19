# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# -----------------------------
# Load trained pipeline
# -----------------------------
if not os.path.exists("pipeline.pkl"):
    raise FileNotFoundError("pipeline.pkl not found. Make sure it exists in the folder.")

pipeline = joblib.load("pipeline.pkl")

# -----------------------------
# Initialize FastAPI
# -----------------------------
app = FastAPI(title="Heart Disease Prediction API")

# -----------------------------
# Request schema
# -----------------------------
class PatientData(BaseModel):
    values: list  # Expecting a list of 13 features from Streamlit

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict(data: PatientData):
    try:
        # Column names must match pipeline training
        columns = [
            "age", "sex", "cp", "trestbps", "chol",
            "fbs", "restecg", "thalach", "exang",
            "oldpeak", "slope", "ca", "thal"
        ]

        # Convert list to DataFrame with correct columns
        df_input = pd.DataFrame([data.values], columns=columns)

        # Make predictions
        pred = int(pipeline.predict(df_input)[0])
        prob = pipeline.predict_proba(df_input)[0].tolist()  # [No Disease, Disease]

        # Return JSON
        return {"prediction": pred, "probability": prob}

    except Exception as e:
        # Return error safely
        return {"prediction": None, "probability": None, "error": str(e)}
