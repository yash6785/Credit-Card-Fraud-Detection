import os
import joblib
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'rf_model.joblib')

def predict_new(sample):
    # Load trained model + scaler
    artifact = joblib.load(MODEL_PATH)
    model = artifact['model']
    scaler = artifact['scaler']

    # Scale input
    sample_scaled = scaler.transform(sample)

    # Predict
    pred = model.predict(sample_scaled)[0]
    proba = model.predict_proba(sample_scaled)[0][1]

    return pred, proba

if __name__ == "__main__":
    # Example transaction (use real transaction features)
    data = pd.DataFrame([[
        -1.359807, -0.072781, 2.536347, 1.378155, -0.338321,
        0.462388, 0.239599, 0.098698, 0.363787, 0.090794,
        -0.551600, -0.617801, -0.991390, -0.311169, 1.468177,
        -0.470400, 0.207971, 0.025791, 0.403993, 0.251412,
        -0.018307, 0.277838, -0.110474, 0.066928, 0.128539,
        -0.189115, 0.133558, -0.021053, 149.62
    ]], columns=[f"V{i}" for i in range(1, 29)] + ["Amount"])

    pred, proba = predict_new(data)
    print("Prediction:", "Fraud" if pred == 1 else "Not Fraud")
    print("Fraud Probability:", round(proba, 4))
