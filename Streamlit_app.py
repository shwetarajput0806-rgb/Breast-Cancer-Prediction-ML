import streamlit as st
import joblib
import numpy as np

# Load Model and Scaler
model = joblib.load("breast_cancer_model.pkl")
scaler = joblib.load("breast_cancer_scaler.pkl") # Load the StandardScaler

# Page Config
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }

    .title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        color: #ff4b4b;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: gray;
        margin-bottom: 30px;
    }

    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        height: 3em;
    }

    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="title">🩺 Breast Cancer Prediction App</p>', unsafe_allow_html=True)

st.markdown(
    '<p class="subtitle">Machine Learning Model using Random Forest Algorithm</p>',
    unsafe_allow_html=True
)

# Input Section
st.sidebar.header("Enter Tumor Details")

radius_mean = st.sidebar.number_input("Radius Mean", min_value=0.0)
texture_mean = st.sidebar.number_input("Texture Mean", min_value=0.0)
perimeter_mean = st.sidebar.number_input("Perimeter Mean", min_value=0.0)
area_mean = st.sidebar.number_input("Area Mean", min_value=0.0)
smoothness_mean = st.sidebar.number_input("Smoothness Mean", min_value=0.0)

# Prediction Button
if st.button("Predict Cancer Type"):

    # Prepare features for scaling and prediction
    raw_features = np.array([[radius_mean,
                          texture_mean,
                          perimeter_mean,
                          area_mean,
                          smoothness_mean]])

    # Scale the input features using the loaded scaler
    features_scaled = scaler.transform(raw_features)

    prediction = model.predict(features_scaled)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.markdown(
            '<div class="prediction-box" style="background-color:#ffcccc; color:#b30000;">⚠ Malignant (Cancerous)</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="prediction-box" style="background-color:#ccffcc; color:#006600;">✅ Benign (Non-Cancerous)</div>',
            unsafe_allow_html=True
        )

# Footer
st.markdown("---")
st.markdown(
    "<center>Developed using Streamlit & Random Forest ML Model</center>",
    unsafe_allow_html=True
)
