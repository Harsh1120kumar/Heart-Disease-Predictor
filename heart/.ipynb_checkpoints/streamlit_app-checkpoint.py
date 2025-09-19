# streamlit_app.py
import streamlit as st
import pandas as pd
import altair as alt
import requests
import time

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>
.stButton>button {
    background-color: #4CAF50;
    color: white;
    font-size:16px;
    border-radius:10px;
}
h1 {
    color: #ff4d4d;
    text-align:center;
}
.footer {
    text-align:center;
    font-size:14px;
    color:gray;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.image("logo.png", width=120)
st.sidebar.markdown("<h4>Developed by Harsh Kumar</h4>", unsafe_allow_html=True)
st.sidebar.write("This app predicts heart disease risk based on patient data.")

# -----------------------------
# HEADER
# -----------------------------
st.image("logo.png", width=150)
st.title("❤️ Heart Disease Prediction Dashboard")
st.markdown("Enter patient data below:")

# -----------------------------
# INPUT FORM
# -----------------------------
with st.form("heart_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 1, 120, 50)
        sex = st.selectbox("Sex", [0,1], format_func=lambda x: "Female" if x==0 else "Male")
        cp = st.selectbox("Chest Pain Type (cp)", [0,1,2,3])
        trestbps = st.number_input("Resting BP (mm Hg)", 80, 250, 120)
        chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)

    with col2:
        fbs = st.selectbox("Fasting Blood Sugar >120 mg/dl", [0,1])
        restecg = st.selectbox("Resting ECG", [0,1,2])
        thalach = st.number_input("Max Heart Rate Achieved", 60, 250, 150)
        exang = st.selectbox("Exercise Induced Angina", [0,1])

    with col3:
        oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0, step=0.1)
        slope = st.selectbox("Slope of Peak ST Segment", [0,1,2])
        ca = st.selectbox("Number of Major Vessels (0-3)", [0,1,2,3])
        thal = st.selectbox("Thalassemia", [1,2,3])

    submitted = st.form_submit_button("Predict")

# -----------------------------
# PREDICTION
# -----------------------------
if submitted:
    # Assemble input in exact order pipeline expects
    input_data_list = [
        int(age), int(sex), int(cp), int(trestbps), int(chol),
        int(fbs), int(restecg), int(thalach), int(exang),
        float(oldpeak), int(slope), int(ca), int(thal)
    ]

    try:
        # Progress bar animation
        st.info("Calculating risk, please wait...")
        progress_bar = st.progress(0)
        for i in range(5):
            time.sleep(0.2)
            progress_bar.progress((i+1)*20)

        # POST to FastAPI
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json={"values": input_data_list}
        )
        result = response.json()

        pred = result.get("prediction")
        prob = result.get("probability")

        if prob is None:
            st.error(result.get("error", "API did not return probability."))
        else:
            # Dynamic background based on prediction
            if pred == 0:
                bg_color = "#e6f7e6"
                st.success("✅ This patient is predicted NOT to have Heart Disease.")
            elif prob[1] >= 0.6:
                bg_color = "#ffe6e6"
                st.error("⚠️ This patient is predicted to have Heart Disease.")
            else:
                bg_color = "#fff9e6"
                st.warning("⚠️ This patient has moderate risk of Heart Disease.")

            st.markdown(
                f"<style>.stApp {{background-color: {bg_color}; transition: background-color 0.5s ease;}}</style>",
                unsafe_allow_html=True
            )

            # Metrics display
            col1, col2 = st.columns(2)
            col1.metric("No Disease Probability", f"{prob[0]*100:.2f}%")
            col2.metric("Disease Probability", f"{prob[1]*100:.2f}%")

            # Probability chart
            prob_df = pd.DataFrame({
                "Class": ["No Disease","Disease"],
                "Probability": prob,
                "Color": ["green","red"]
            })

            chart = alt.Chart(prob_df).mark_bar().encode(
                x="Class",
                y=alt.Y("Probability", scale=alt.Scale(domain=[0,1])),
                color=alt.Color("Color", scale=None),
                tooltip=["Class", alt.Tooltip("Probability", format=".2f")]
            ).properties(width=500, height=400, title="Prediction Probability")

            st.altair_chart(chart, use_container_width=True)

    except Exception as e:
        st.error(f"Error connecting to API: {e}")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown(
    '<div class="footer">❤️ Made by Harsh Kumar | Powered by Streamlit & FastAPI</div>',
    unsafe_allow_html=True
)
