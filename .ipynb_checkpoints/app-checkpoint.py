from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Load model and encoder
try:
    model = joblib.load("heart_disease_rf_model.pkl")
    encoder = joblib.load("encoder.pkl")
except FileNotFoundError:
    model = None
    encoder = None

# Define categorical columns (same as during training)
categorical_cols = ['cp','slope','thal']
all_features = ['age','sex','cp','trestbps','chol','fbs',
                'restecg','thalach','exang','oldpeak',
                'slope','ca','thal']

# FastAPI app
app = FastAPI()

# Input schema
class HeartInput(BaseModel):
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

@app.post("/predict")
def predict_heart_disease(input_data: HeartInput):
    if model is None or encoder is None:
        return {"error": "Model or encoder file not found. Please train and save first."}

    # Convert input to DataFrame
    input_dict = input_data.dict()
    input_df = pd.DataFrame([input_dict], columns=all_features)

    # Separate numeric and categorical
    numeric_df = input_df.drop(columns=categorical_cols)
    encoded = encoder.transform(input_df[categorical_cols])

    # Combine numeric + encoded
    final_input = np.hstack([numeric_df.to_numpy(), encoded])

    # Predict
    prediction = model.predict(final_input)[0]
    probability = model.predict_proba(final_input)[0][1]

    return {
        "prediction": int(prediction),  # 0 = No Disease, 1 = Disease
        "probability": float(probability)
    }
