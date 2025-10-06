# data_preprocess.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def load_data(path='data/HTRU_2.csv'):
    """
    Load HTRU_2.csv dataset.
    Assumes last column is the class label (0 or 1).
    Returns: X (numpy array), y (numpy array)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. Please place HTRU_2.csv in the data/ folder.")
    df = pd.read_csv(path, header=None)
    # last column is class
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def split_and_scale(X, y, test_size=0.2, random_state=42, save_scaler=True):
    """
    Stratified train/test split and standard scaling.
    Saves scaler to results/models/scaler.pkl if save_scaler=True.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if save_scaler:
        os.makedirs('results/models', exist_ok=True)
        joblib.dump(scaler, 'results/models/scaler.pkl')
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_and_scale(X, y)
    print("Done preprocessing.")
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
    print("Positive labels in train:", int(y_train.sum()), "in test:", int(y_test.sum()))
