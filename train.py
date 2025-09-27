import os, joblib
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from src.data_processing import load_data, preprocess

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

def train_model():
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess(df)
    
    smote = SMOTE(random_state=42)
    model = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    pipeline = Pipeline([("smote", smote), ("clf", model)])
    
    pipeline.fit(X_train, y_train)
    
    joblib.dump({"model": pipeline, "scaler": scaler}, os.path.join(MODEL_DIR, "rf_model.joblib"))
    print("✅ Model trained & saved in models/rf_model.joblib")

if __name__ == "__main__":
    train_model()
