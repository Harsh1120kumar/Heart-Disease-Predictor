# ❤️ Heart Disease Prediction App

## Overview
This project is a **Heart Disease Prediction System** that allows users to predict the risk of heart disease based on patient data. It features a **user-friendly frontend built with Streamlit** and a **machine learning backend powered by FastAPI**. The app provides a **probability-based prediction**, visual charts, and dynamic interface cues based on risk level.

---

## Features
- Predicts the presence or absence of heart disease in patients.
- Displays **probability metrics** for “No Disease” and “Disease” classes.
- Dynamic **color-coded interface** to indicate risk:
  - Soft green for low risk
  - Soft yellow for moderate risk
  - Gentle red for high risk
- **Interactive probability chart** using Altair.
- Fully integrated **frontend (Streamlit) and backend (FastAPI)**.
- Can be easily deployed and shared.

---

## How It Works

### 1️⃣ Data Collection & Preprocessing
- Input features: `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`.
- **Categorical features** (`cp`, `thal`, `slope`) are one-hot encoded.
- **Numerical features** are standardized using `StandardScaler`.
- All preprocessing steps are included in a **trained pipeline**.

### 2️⃣ Machine Learning Model
- **Random Forest Classifier** is used for prediction.
- Hyperparameter tuning is applied to improve accuracy.
- The trained model and preprocessing steps are stored in `pipeline.pkl`.

### 3️⃣ Backend (FastAPI)
- Handles POST requests from the frontend.
- Accepts input as a **list of 13 features**.
- Converts the list into a **DataFrame** to match the pipeline.
- Returns **prediction** and **probability** JSON.

### 4️⃣ Frontend (Streamlit)
- User inputs patient data through a **clean, interactive interface**.
- Shows dynamic **risk-based background** and **metrics**.
- Displays **bar chart** of prediction probabilities.
- Provides **progress animation** while calculating risk.

