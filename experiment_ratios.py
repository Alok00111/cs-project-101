import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")

# Insert path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import load_data, preprocess_data
from train_models import build_models
from utils import TARGET_COLUMN

def run_experiment():
    print("Loading and preprocessing data...")
    df_raw = load_data()
    # Suppress console spam from preprocess_data by replacing sys.stdout temporarily if needed, 
    # but let's just let it print for now.
    df_clean, encoders, feature_cols = preprocess_data(df_raw)

    X = df_clean[feature_cols].values
    y = df_clean[TARGET_COLUMN].values

    test_sizes = [0.4, 0.3, 0.25, 0.2, 0.15, 0.1] # corresponding to 60-40, 70-30, 75-25, 80-20, 85-15, 90-10
    
    best_overall_acc = 0
    best_overall_model = None
    best_overall_ratio = None
    best_overall_split = ""

    print("\nStarting Ratio Experiments...")
    
    for test_size in test_sizes:
        train_pct = int(100 * (1 - test_size))
        test_pct = int(100 * test_size)
        split_name = f"{train_pct}-{test_pct}"
        print(f"\n======================================")
        print(f"Testing Split Ratio: {split_name} (Test Size: {test_size})")
        print(f"======================================")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        models = build_models()
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            print(f"  {name:20s}: {acc:.4f}")
            
            if acc > best_overall_acc:
                best_overall_acc = acc
                best_overall_model = name
                best_overall_ratio = test_size
                best_overall_split = split_name

    print("\n" + "="*50)
    print("              EXPERIMENT RESULTS")
    print("="*50)
    print(f"BEST SPLIT RATIO: {best_overall_split} (Test Size: {best_overall_ratio})")
    print(f"BEST MODEL:       {best_overall_model}")
    print(f"BEST ACCURACY:    {best_overall_acc:.4f}")
    print("="*50)

if __name__ == "__main__":
    run_experiment()
