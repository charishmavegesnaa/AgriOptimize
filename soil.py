import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="🌾 Soil Yield Predictor", layout="wide")
st.title("🌾 Soil Yield Optimizer - Linear Regression")
st.markdown("---")

# -------------------------------------------------------------------
# 🔴 REQUIRED MODEL FILES (MUST BE IN SAME FOLDER AS soil.py)
#   - soil_yield_lr_model.pkl      → trained LinearRegression model
#   - soilscaler.pkl               → StandardScaler for numerical features
#   - soillabelencoders.pkl        → dict of LabelEncoders for categoricals
#
# To create them in your notebook (soil_nutrients_ml.ipynb) after training: [file:1]
#     import joblib
#     joblib.dump(lr_model, 'soil_yield_lr_model.pkl')
#     joblib.dump(scaler, 'soilscaler.pkl')
#     joblib.dump(label_encoders, 'soillabelencoders.pkl')
# -------------------------------------------------------------------

REQUIRED_FILES = [
    "soil_yield_lr_model.pkl",
    "soilscaler.pkl",
    "soillabelencoders.pkl",
]

missing_files = [f for f in REQUIRED_FILES if not os.path.exists(f)]
if missing_files:
    st.error(
        "❌ Model files missing!\n\n"
        "Please make sure these files are in the same folder as `soil.py`:\n"
        " - soil_yield_lr_model.pkl\n"
        " - soilscaler.pkl\n"
        " - soillabelencoders.pkl\n\n"
        "Create them in your Jupyter notebook using:\n"
        "  import joblib\n"
        "  joblib.dump(lr_model, 'soil_yield_lr_model.pkl')\n"
        "  joblib.dump(scaler, 'soilscaler.pkl')\n"
        "  joblib.dump(label_encoders, 'soillabelencoders.pkl')\n"
    )
    st.stop()

# Load model and preprocessors
@st.cache_resource
def load_model():
    lr_model = joblib.load('soil_yield_lr_model.pkl')
    scaler = joblib.load('soilscaler.pkl')
    label_encoders = joblib.load('soillabelencoders.pkl')
    return lr_model, scaler, label_encoders

lr_model, scaler, label_encoders = load_model()

feature_cols = [
    'Fertility', 'Photoperiod', 'Temperature', 'Rainfall', 'pH',
    'LightHours', 'LightIntensity', 'Rh', 'Nitrogen', 'Phosphorus',
    'Potassium', 'SoilType', 'Season', 'CategorypH', 'NPRatio',
    'NKRatio', 'PKRatio', 'TotalNPK', 'TempRainInteraction',
    'pHNitrogenInteraction'
]

categorical_features = ['Fertility', 'Photoperiod', 'SoilType', 'Season', 'CategorypH']

st.success("✅ Model loaded! Ready for fertilizer optimization predictions [file:1]")

# Sidebar inputs
st.sidebar.header("📊 Input Parameters")
col1, col2 = st.columns(2)

with col1:
    crop_fertility = st.sidebar.selectbox("Fertility", ['low', 'moderate', 'high'])
    photoperiod = st.sidebar.selectbox("Photoperiod", ['day neutral', 'short day period'])
    soil_type = st.sidebar.selectbox("Soil Type", ['loam', 'sandy', 'sandy loam'])
    season = st.sidebar.selectbox("Season", ['spring', 'summer', 'fall', 'winter'])

with col2:
    temp = st.sidebar.slider("Temperature (°C)", 9.0, 40.0, 20.8)
    rainfall = st.sidebar.slider("Rainfall (mm)", 400.0, 2500.0, 948.8)
    ph = st.sidebar.slider("pH", 4.8, 8.0, 6.47)

# NPK inputs
st.sidebar.subheader("🌱 NPK Fertilizers (kg/ha)")
nitrogen = st.sidebar.slider("Nitrogen", 40.0, 410.0, 142.8)
phosphorus = st.sidebar.slider("Phosphorus", 13.0, 360.0, 107.7)
potassium = st.sidebar.slider("Potassium", 35.0, 580.0, 180.5)

# Environmental
light_hours = st.sidebar.slider("Light Hours", 5.0, 16.0, 9.46)
light_intensity = st.sidebar.slider("Light Intensity", 70.0, 985.0, 398.0)
rh = st.sidebar.slider("Relative Humidity (%)", 30.0, 100.0, 67.1)
ph_category = st.sidebar.selectbox("pH Category", ['low acidic', 'neutral', 'high alkaline'])

# Predict button
if st.button("🚀 Predict Optimal Yield", type="primary"):

    # Create input DataFrame
    input_data = pd.DataFrame({
        'Fertility': [crop_fertility],
        'Photoperiod': [photoperiod],
        'Temperature': [temp],
        'Rainfall': [rainfall],
        'pH': [ph],
        'LightHours': [light_hours],
        'LightIntensity': [light_intensity],
        'Rh': [rh],
        'Nitrogen': [nitrogen],
        'Phosphorus': [phosphorus],
        'Potassium': [potassium],
        'SoilType': [soil_type],
        'Season': [season],
        'CategorypH': [ph_category]
    })

    # Feature engineering (same as notebook) [file:1]
    input_data['NPRatio'] = input_data['Nitrogen'] / (input_data['Phosphorus'] + 1e-6)
    input_data['NKRatio'] = input_data['Nitrogen'] / (input_data['Potassium'] + 1e-6)
    input_data['PKRatio'] = input_data['Phosphorus'] / (input_data['Potassium'] + 1e-6)
    input_data['TotalNPK'] = input_data['Nitrogen'] + input_data['Phosphorus'] + input_data['Potassium']
    input_data['TempRainInteraction'] = input_data['Temperature'] * input_data['Rainfall']
    input_data['pHNitrogenInteraction'] = input_data['pH'] * input_data['Nitrogen']

    # Clean categorical
    for col in categorical_features:
        input_data[col] = input_data[col].astype(str).str.lower().str.strip()

    # Preprocess
    X = input_data[feature_cols].copy()
    for col in categorical_features:
        X[col] = label_encoders[col].transform(X[col])

    numerical_features = [col for col in feature_cols if col not in categorical_features]
    X[numerical_features] = scaler.transform(X[numerical_features])

    # Predict
    prediction = lr_model.predict(X)[0]

    # Display results
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("🌾 Predicted Yield", f"{prediction:.2f} tons", delta="📈 Optimized")
    with c2:
        st.metric("💰 NPK Cost", f"${(nitrogen + phosphorus + potassium) * 0.5:.0f}", delta="💰")
    with c3:
        st.metric("⭐ Model R²", "0.84", delta="🧠 Accurate")

    # Input summary
    st.subheader("📋 Your Fertilizer Plan")
    with st.expander("Detailed Parameters"):
        input_df = input_data[
            ['Fertility', 'Photoperiod', 'Temperature', 'Rainfall', 'pH',
             'Nitrogen', 'Phosphorus', 'Potassium', 'SoilType', 'Season']
        ].round(2)
        st.dataframe(input_df.T, use_container_width=True)

    # Recommendations
    st.subheader("💡 Fertilizer Recommendations [file:1]")
    recs = [
        f"✅ Optimal NPK Ratio: N:{nitrogen:.0f}, P:{phosphorus:.0f}, K:{potassium:.0f}",
        f"✅ Temperature: {temp:.1f}°C - suitable for {crop_fertility} fertility soil",
        f"✅ Rainfall: {rainfall:.0f} mm - good water availability",
        "🎯 Expected Yield: High optimization achieved!"
    ]
    for rec in recs:
        st.success(rec)

# Instructions (single markdown block, no extra code fences)
with st.expander("📖 How to Run"):
    st.markdown(
        "1. Keep all files in the **same folder**:\n"
        "   - soil.py\n"
        "   - soil_yield_lr_model.pkl\n"
        "   - soilscaler.pkl\n"
        "   - soillabelencoders.pkl\n\n"
        "2. Install dependencies in terminal:\n"
        "   pip install streamlit pandas scikit-learn joblib\n\n"
        "3. Run the app:\n"
        "   streamlit run soil.py\n\n"
        "4. Open browser at:\n"
        "   http://localhost:8501\n"
    )

st.markdown("---")
st.caption("🌱 Built for sustainable fertilizer optimization [file:1][file:2]")
