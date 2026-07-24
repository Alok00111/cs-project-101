"""
prediction.py
-------------
Reusable prediction module for the CDSS Diabetes Mellitus pipeline.
Supports any of the four saved models and is designed to be imported
directly into a Flask / FastAPI backend.
"""

import os
import numpy as np
import joblib

from utils import (
    LR_MODEL_PATH,
    DT_MODEL_PATH,
    RF_MODEL_PATH,
    XGB_MODEL_PATH,
    ENCODERS_PATH,
    FEATURES_PATH,
    SCALER_PATH,
    TARGET_COLUMN,
    logger,
)
from preprocessing import preprocess_single_patient


# ---------------------------------------------------------------------------
# MODEL REGISTRY
# ---------------------------------------------------------------------------
MODEL_PATHS = {
    "logistic_regression": LR_MODEL_PATH,
    "decision_tree":       DT_MODEL_PATH,
    "random_forest":       RF_MODEL_PATH,
    "xgboost":             XGB_MODEL_PATH,
}


def _load_artifact(path: str):
    """Load a joblib artifact with a helpful error message."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Artifact not found: {path}\n"
            "Run train_models.py first to generate trained models."
        )
    return joblib.load(path)


# =============================================================================
# MAIN PREDICTION FUNCTION
# =============================================================================
def predict_patient(
    patient_data: dict,
    model_name: str = "random_forest",
) -> dict:
    """
    Predict the readmission class for a single patient.

    Parameters
    ----------
    patient_data : dict
        Raw patient data as {column_name: value} pairs.
        Missing columns will be filled with defaults automatically.
    model_name : str
        One of: 'logistic_regression', 'decision_tree',
                'random_forest', 'xgboost'.
        Defaults to 'random_forest' (typically best performer).

    Returns
    -------
    dict with keys:
        predicted_class   : str   — decoded label (e.g., 'NO', '>30', '<30')
        predicted_index   : int   — numeric class index
        class_probabilities : dict — {class_label: probability}
        confidence        : float — probability of the predicted class
        model_used        : str
    """
    # --- 1. Validate model name -------------------------------------------
    model_name = model_name.lower().replace(" ", "_")
    if model_name not in MODEL_PATHS:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {list(MODEL_PATHS.keys())}"
        )

    # --- 2. Load artifacts -----------------------------------------------
    model    = _load_artifact(MODEL_PATHS[model_name])
    encoders = _load_artifact(ENCODERS_PATH)

    # --- 3. Preprocess patient data --------------------------------------
    X = preprocess_single_patient(patient_data)

    # --- 4. Scale if scaler exists ----------------------------------------
    if os.path.exists(SCALER_PATH):
        scaler = _load_artifact(SCALER_PATH)
        X = scaler.transform(X)

    # --- 5. Predict -------------------------------------------------------
    y_index = int(model.predict(X)[0])

    # --- 6. Decode target label -------------------------------------------
    target_encoder = encoders.get(TARGET_COLUMN)
    if target_encoder is not None:
        predicted_class = target_encoder.inverse_transform([y_index])[0]
        class_labels    = list(target_encoder.classes_)
    else:
        predicted_class = str(y_index)
        class_labels    = [str(i) for i in range(model.n_classes_ if hasattr(model, "n_classes_") else 3)]

    # --- 7. Probabilities -------------------------------------------------
    class_probabilities = {}
    confidence = 1.0

    try:
        proba = model.predict_proba(X)[0]
        confidence = float(proba[y_index])
        class_probabilities = {
            label: round(float(p), 6)
            for label, p in zip(class_labels, proba)
        }
    except AttributeError:
        class_probabilities = {predicted_class: 1.0}

    return {
        "predicted_class":       predicted_class,
        "predicted_index":       y_index,
        "class_probabilities":   class_probabilities,
        "confidence":            round(confidence, 6),
        "model_used":            model_name,
    }


# =============================================================================
# CONVENIENCE: predict from all models (for comparison)
# =============================================================================
def predict_all_models(patient_data: dict) -> dict:
    """
    Run all four models on the same patient and return results dict.

    Returns
    -------
    dict  {model_name: predict_patient(...) result}
    """
    all_results = {}
    for model_name in MODEL_PATHS:
        try:
            all_results[model_name] = predict_patient(patient_data, model_name)
        except Exception as exc:
            all_results[model_name] = {"error": str(exc)}
    return all_results


# =============================================================================
# QUICK TEST (run directly)
# =============================================================================
if __name__ == "__main__":
    # Example dummy patient — column names should match your dataset features
    dummy_patient = {
        "age":                "[50-60)",
        "gender":             "Male",
        "race":               "Caucasian",
        "time_in_hospital":   5,
        "num_lab_procedures": 40,
        "num_procedures":     1,
        "num_medications":    15,
        "number_outpatient":  0,
        "number_emergency":   0,
        "number_inpatient":   1,
        "number_diagnoses":   7,
        "A1Cresult":          ">7",
        "max_glu_serum":      "None",
        "insulin":            "Yes",
        "diabetesMed":        "Yes",
        "change":             "Ch",
    }

    print("Running prediction on dummy patient...")
    result = predict_patient(dummy_patient, model_name="random_forest")
    print("\nPrediction Result:")
    for k, v in result.items():
        print(f"  {k}: {v}")
