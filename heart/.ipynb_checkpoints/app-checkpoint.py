# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
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
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict(data: PatientData):
    try:
        # Convert input to list
        input_values = [list(data.dict().values())]

        # Make predictions
        pred = pipeline.predict(input_values)[0]
        prob = pipeline.predict_proba(input_values)[0].tolist()  # [No Disease, Disease]

        # Return safe JSON
        return {"prediction": int(pred), "probability": prob}

    except Exception as e:
        # Return error JSON for Streamlit to handle safely
        return {"prediction": None, "probability": None, "error": str(e)}
