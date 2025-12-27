import streamlit as st
import numpy as np
import joblib

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
        model = joblib.load("soil_yield_lr_model.pkl")
        scaler = joblib.load("soil_scaler.pkl")
        label_encoders = joblib.load("soil_label_encoders.pkl")
        return model, scaler, label_encoders
    except FileNotFoundError:
        st.error(
            "❌ Model files missing! Need:\n"
            "- soil_yield_lr_model.pkl\n"
            "- soil_scaler.pkl\n"
            "- soil_label_encoders.pkl"
        )
        st.stop()

model, scaler, label_encoders = load_model()

# -------------------------------
# Detect correct encoder keys
# -------------------------------
encoder_keys = list(label_encoders.keys())

# Try to auto-detect keys safely
soil_key = next(k for k in encoder_keys if "soil" in k.lower())
crop_key = next(k for k in encoder_keys if "crop" in k.lower())

# -------------------------------
# Input Section
# -------------------------------
st.subheader("🧪 Enter Soil Parameters")

soil_type = st.selectbox(
    "Soil Type",
    label_encoders[soil_key].classes_
)

crop_type = st.selectbox(
    "Crop Type",
    label_encoders[crop_key].classes_
)

nitrogen = st.number_input("Nitrogen (N)", min_value=0.0)
phosphorus = st.number_input("Phosphorus (P)", min_value=0.0)
potassium = st.number_input("Potassium (K)", min_value=0.0)
ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, step=0.1)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0)

# -------------------------------
# Prediction
# -------------------------------
if st.button("🌱 Predict Yield"):
    try:
        soil_encoded = label_encoders[soil_key].transform([soil_type])[0]
        crop_encoded = label_encoders[crop_key].transform([crop_type])[0]

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
        prediction = model.predict(input_scaled)[0]

        st.success(f"🌾 Estimated Crop Yield: **{prediction:.2f} tons/hectare**")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Linear Regression based Soil Yield Prediction System")
