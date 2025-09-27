import streamlit as st
import pandas as pd
import joblib, os

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'rf_model.joblib')

st.title("💳 Credit Card Fraud Detection App")

uploaded = st.file_uploader("Upload a CSV with transactions", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    scaler = artifact["scaler"]
    
    if "Time" in df.columns:
        df = df.drop(columns=["Time"])
    proba = model.predict_proba(df)[:,1]
    
    df["fraud_probability"] = proba
    df["predicted"] = (df["fraud_probability"] > 0.5).astype(int)
    
    st.write("Predictions:")
    st.dataframe(df.head())
