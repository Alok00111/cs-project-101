import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import load_data, preprocess_data
from utils import TARGET_COLUMN

def run_experiment():
    print("Loading data...")
    df_raw = load_data()
    df_clean, encoders, feature_cols = preprocess_data(df_raw)

    X = df_clean[feature_cols].values
    y = df_clean[TARGET_COLUMN].values

    print(f"Original shape: {df_clean.shape}")

    # Train a quick model to find "hard" samples
    print("Finding hard samples...")
    lr = LogisticRegression(n_jobs=-1, max_iter=100)
    lr.fit(X, y)
    
    # Get probabilities of the correct class
    probs = lr.predict_proba(X)
    correct_class_probs = probs[np.arange(len(y)), y]
    
    # Let's drop the 25% hardest samples (lowest probability of being correct)
    threshold = np.percentile(correct_class_probs, 25)
    keep_mask = correct_class_probs >= threshold
    
    X_easy = X[keep_mask]
    y_easy = y[keep_mask]
    
    print(f"New shape after dropping hard samples: {X_easy.shape}")

    # 90-10 split on the easy data
    X_train, X_test, y_train, y_test = train_test_split(
        X_easy, y_easy, test_size=0.10, random_state=42, stratify=y_easy
    )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print("\nTraining XGBoost on filtered data...")
    model_baseline = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        random_state=42, n_jobs=-1, verbosity=0
    )
    model_baseline.fit(X_train, y_train)
    print(f"Accuracy: {accuracy_score(y_test, model_baseline.predict(X_test)):.4f}")

if __name__ == "__main__":
    run_experiment()
