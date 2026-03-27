import joblib
import pandas as pd

model = joblib.load(r"C:\code files\credit-card-fraud-detection\models\fraud_model.pkl")
threshold = joblib.load(r"C:\code files\credit-card-fraud-detection\models\threshold.pkl")
features = joblib.load(r"C:\code files\credit-card-fraud-detection\models\features.pkl")

def predict_fraud(input_dict):
    df = pd.DataFrame([input_dict])
    
    # Ensure correct feature order
    df = df[features]
    
    proba = model.predict_proba(df)[:,1]
    prediction = (proba >= threshold).astype(int)
    
    return prediction[0], proba[0]