import streamlit as st
import joblib
import pandas as pd

# Load artifacts
model = joblib.load(r"C:\code files\credit-card-fraud-detection\models\fraud_model.pkl")
threshold = joblib.load(r"C:\code files\credit-card-fraud-detection\models\threshold.pkl")
features = joblib.load(r"C:\code files\credit-card-fraud-detection\models\features.pkl")

st.title("💳 Credit Card Fraud Detection")

st.write("Enter transaction details to check if it's fraud.")

# Create input fields dynamically
input_data = {}

for feature in features:
    input_data[feature] = st.number_input(f"{feature}", value=0.0)

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Prediction button
if st.button("Predict"):
    proba = model.predict_proba(input_df)[:,1]
    prediction = (proba >= threshold).astype(int)

    st.subheader("Result:")
    
    if prediction[0] == 1:
        st.error(f"⚠️ Fraud Detected! Probability: {proba[0]:.4f}")
    else:
        st.success(f"✅ Legitimate Transaction. Probability: {proba[0]:.4f}")