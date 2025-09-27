import os, joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from src.data_processing import load_data, preprocess

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'rf_model.joblib')

def evaluate():
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess(df)
    
    artifact = joblib.load(MODEL_PATH)
    model = artifact['model']
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    
    print("📊 Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))

if __name__ == "__main__":
    evaluate()
