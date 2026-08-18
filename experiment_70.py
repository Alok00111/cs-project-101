import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import warnings

# Try to import LightGBM and CatBoost
try:
    from lightgbm import LGBMClassifier
    has_lgb = True
except ImportError:
    has_lgb = False

try:
    from catboost import CatBoostClassifier
    has_cat = True
except ImportError:
    has_cat = False

try:
    from imblearn.over_sampling import SMOTE
    has_smote = True
except ImportError:
    has_smote = False

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

    # 90-10 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.10, random_state=42, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print("\n" + "-"*40)
    print("1. Baseline XGBoost (Current)")
    model_baseline = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        random_state=42, n_jobs=-1, verbosity=0
    )
    model_baseline.fit(X_train, y_train)
    print(f"Accuracy: {accuracy_score(y_test, model_baseline.predict(X_test)):.4f}")

    print("\n" + "-"*40)
    print("2. Tuned XGBoost (More trees, deeper, smaller LR)")
    model_tuned = XGBClassifier(
        n_estimators=500, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        random_state=42, n_jobs=-1, verbosity=0
    )
    model_tuned.fit(X_train, y_train)
    print(f"Accuracy: {accuracy_score(y_test, model_tuned.predict(X_test)):.4f}")

    if has_lgb:
        print("\n" + "-"*40)
        print("3. LightGBM")
        model_lgb = LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=63,
            random_state=42, n_jobs=-1, verbose=-1
        )
        model_lgb.fit(X_train, y_train)
        print(f"Accuracy: {accuracy_score(y_test, model_lgb.predict(X_test)):.4f}")
    
    if has_cat:
        print("\n" + "-"*40)
        print("4. CatBoost")
        model_cat = CatBoostClassifier(
            iterations=500, learning_rate=0.05, depth=8,
            random_state=42, verbose=0, thread_count=-1
        )
        model_cat.fit(X_train, y_train)
        print(f"Accuracy: {accuracy_score(y_test, model_cat.predict(X_test)):.4f}")

    if has_smote:
        print("\n" + "-"*40)
        print("5. SMOTE + Tuned XGBoost")
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        
        model_smote = XGBClassifier(
            n_estimators=500, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=42, n_jobs=-1, verbosity=0
        )
        model_smote.fit(X_train_res, y_train_res)
        print(f"Accuracy: {accuracy_score(y_test, model_smote.predict(X_test)):.4f}")

if __name__ == "__main__":
    run_experiment()
