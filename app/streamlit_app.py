import sys
import os

sys.path.append(r"C:\code files\credit-card-fraud-detection")

import streamlit as st
import joblib
import pandas as pd

from src.predict import predict_fraud

# Page config
st.set_page_config(page_title="Fraud Detection", layout="centered")


st.title("💳 Credit Card Fraud Detection")
st.markdown("Detect fraudulent transactions using a trained Machine Learning model")

st.divider()


features = joblib.load(r"C:\code files\credit-card-fraud-detection\models\features.pkl")

# Input section
st.subheader("🔢 Enter Transaction Details")

input_data = {}


col1, col2 = st.columns(2)

for i, feature in enumerate(features):
    if i % 2 == 0:
        input_data[feature] = col1.number_input(feature, value=0.0)
    else:
        input_data[feature] = col2.number_input(feature, value=0.0)

st.divider()

# Prediction
if st.button("🚀 Predict"):
    prediction, proba = predict_fraud(input_data)

    st.subheader("📊 Prediction Result")

    st.metric("Fraud Probability", f"{proba*100:.2f}%")

    if prediction == 1:
        st.error("⚠️ Fraudulent Transaction Detected")
    else:
        st.success("✅ Legitimate Transaction")

st.divider()

st.caption("Model: Tuned Random Forest | Threshold Optimized")