import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Soil Yield Optimizer - Linear Regression",
    page_icon="🌾",
    layout="centered"
)

st.title("🌾 Soil Yield Optimizer - Linear Regression")
st.markdown("---")

# -------------------------------
# Load model files
# -------------------------------
@st.cache_resource
def load_model():
    try:
        lr_model = joblib.load("soil_yield_lr_model.pkl")
        scaler = joblib.load("soil_scaler.pkl")
        label_encoders = joblib.load("soil_label_encoders.pkl")
        return lr_model, scaler, label_encoders
    except FileNotFoundError:
        st.error(
            "❌ Model files missing! Need: "
            "soil_yield_lr_model.pkl, soil_scaler.pkl, soil_label_encoders.pkl"
        )
        st.stop()

lr_model, scaler, label_encoders = load_model()

# -------------------------------
# User Input Section
# -------------------------------
st.subheader("🧪 Enter Soil Parameters")

soil_type = st.selectbox(
    "Soil Type",
    label_encoders["Soil_Type"].classes_
)

crop_type = st.selectbox(
    "Crop Type",
    label_encoders["Crop"].classes_
)

nitrogen = st.number_input("Nitrogen (N)", min_value=0.0, step=1.0)
phosphorus = st.number_input("Phosphorus (P)", min_value=0.0, step=1.0)
potassium = st.number_input("Potassium (K)", min_value=0.0, step=1.0)
ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, step=0.1)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, step=1.0)

# -------------------------------
# Prediction
# -------------------------------
if st.button("🌱 Predict Yield"):
    try:
        soil_encoded = label_encoders["Soil_Type"].transform([soil_type])[0]
        crop_encoded = label_encoders["Crop"].transform([crop_type])[0]

        input_data = np.array([[
            soil_encoded,
            crop_encoded,
            nitrogen,
            phosphorus,
            potassium,
            ph,
            rainfall
        ]])

        input_scaled = scaler.transform(input_data)
        prediction = lr_model.predict(input_scaled)[0]

        st.success(f"🌾 Estimated Crop Yield: **{prediction:.2f} tons/hectare**")

    except Exception as e:
        st.error(f"Prediction error: {e}")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Linear Regression based Soil Yield Prediction System")
