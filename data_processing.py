import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'creditcard.csv')

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df

def preprocess(df, test_size=0.2, random_state=42):
    X = df.drop(columns=['Class', 'Time'])
    y = df['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler
