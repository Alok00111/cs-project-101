"""
train_models.py
---------------
Master execution script for the CDSS Diabetes Mellitus ML pipeline.

Run with:
    python train_models.py

Stages:
    1.  Load data
    2.  Preprocess
    3.  EDA visualizations
    4.  Train / Test split
    5.  Train models
    6.  Evaluate models
    7.  Generate result visualizations
    8.  Build comparison table
    9.  Save models
    10. Print final summary
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd

# Make sure sibling modules are importable when run from any working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.linear_model  import LogisticRegression
from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import RandomForestClassifier
from xgboost               import XGBClassifier

from utils          import (
    setup_directories,
    section_header,
    print_dict_table,
    start_timer,
    elapsed,
    GRAPHS_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    LR_MODEL_PATH,
    DT_MODEL_PATH,
    RF_MODEL_PATH,
    XGB_MODEL_PATH,
    RANDOM_STATE,
    logger,
)
from preprocessing  import load_data, preprocess_data, split_data
from evaluation     import evaluate_model, build_comparison_table, get_best_worst
from visualization  import (
    plot_missing_value_heatmap,
    plot_missing_value_summary,
    plot_target_distribution,
    plot_age_distribution,
    plot_gender_distribution,
    plot_race_distribution,
    plot_num_diagnoses,
    plot_num_medications,
    plot_hba1c_distribution,
    plot_glucose_distribution,
    plot_hospital_stay,
    plot_correlation_heatmap,
    plot_all_metric_comparisons,
    plot_time_comparisons,
    plot_roc_curves,
    plot_all_confusion_matrices,
    plot_decision_tree,
    plot_feature_importance,
)


# =============================================================================
# GLOBAL PIPELINE TIMER
# =============================================================================
PIPELINE_START = start_timer()


# =============================================================================
# STEP 5 — MODEL DEFINITIONS
# =============================================================================
def build_models() -> dict:
    """
    Return a dict of {name: unfitted_model} with sensible hyperparameters.
    All models support multiclass classification.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            C=1.0,
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            criterion="gini",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        ),
    }


# =============================================================================
# STEP 5 & 6 — TRAIN + EVALUATE
# =============================================================================
def train_and_evaluate(
    models: dict,
    X_train: np.ndarray,
    X_test:  np.ndarray,
    y_train: np.ndarray,
    y_test:  np.ndarray,
    feature_cols: list,
    encoders: dict,
    target_col: str = "readmitted",
) -> dict:
    """
    Train every model, measure timing, evaluate, and return results dict.
    """
    results    = {}
    model_path_map = {
        "Logistic Regression": LR_MODEL_PATH,
        "Decision Tree":       DT_MODEL_PATH,
        "Random Forest":       RF_MODEL_PATH,
        "XGBoost":             XGB_MODEL_PATH,
    }

    # Decode class names for display
    target_encoder = encoders.get(target_col)
    if target_encoder is not None:
        class_names = list(target_encoder.classes_)
    else:
        class_names = [str(c) for c in np.unique(y_test)]

    for name, model in models.items():
        section_header(f"STEP 5 — Training: {name}")

        # --- Train ----------------------------------------------------------
        t0 = start_timer()
        model.fit(X_train, y_train)
        t_train = elapsed(t0)
        logger.info(f"Training complete in {t_train:.4f}s")

        # --- Predict --------------------------------------------------------
        t1 = start_timer()
        _ = model.predict(X_test)
        t_pred = elapsed(t1)
        logger.info(f"Prediction complete in {t_pred:.4f}s")

        # --- Evaluate -------------------------------------------------------
        result = evaluate_model(
            name=name,
            model=model,
            X_test=X_test,
            y_test=y_test,
            X_train=X_train,
            y_train=y_train,
            train_time=t_train,
            pred_time=t_pred,
            class_names=class_names,
        )
        results[name] = result

        # --- Save model -----------------------------------------------------
        save_path = model_path_map.get(name)
        if save_path:
            joblib.dump(model, save_path)
            logger.info(f"Model saved to: {save_path}")

    return results


# =============================================================================
# STEP 7 — VISUALIZATIONS (post-training)
# =============================================================================
def generate_result_visualizations(
    results: dict,
    models: dict,
    feature_cols: list,
    encoders: dict,
    target_col: str = "readmitted",
) -> None:
    section_header("STEP 7 — Generating Result Visualizations")

    # Decode class names
    target_encoder = encoders.get(target_col)
    class_names = list(target_encoder.classes_) if target_encoder else None

    # Metric comparison bar charts
    plot_all_metric_comparisons(results)

    # Timing comparison
    plot_time_comparisons(results)

    # ROC curves
    plot_roc_curves(results)

    # Confusion matrices (one per model)
    plot_all_confusion_matrices(results)

    # Decision Tree visualization
    dt_model = models.get("Decision Tree")
    if dt_model is not None:
        try:
            plot_decision_tree(dt_model, feature_cols, class_names or [])
        except Exception as exc:
            logger.warning(f"Decision Tree plot failed: {exc}")

    # Random Forest feature importance
    rf_model = models.get("Random Forest")
    if rf_model is not None and hasattr(rf_model, "feature_importances_"):
        plot_feature_importance(
            rf_model.feature_importances_,
            feature_cols,
            "Random Forest",
            "25_rf",
        )

    # XGBoost feature importance
    xgb_model = models.get("XGBoost")
    if xgb_model is not None and hasattr(xgb_model, "feature_importances_"):
        plot_feature_importance(
            xgb_model.feature_importances_,
            feature_cols,
            "XGBoost",
            "26_xgb",
        )


# =============================================================================
# STEP 12 — FINAL SUMMARY
# =============================================================================
def print_final_summary(
    df: pd.DataFrame,
    X_train: np.ndarray,
    X_test:  np.ndarray,
    results: dict,
    comparison_df: pd.DataFrame,
) -> None:
    section_header("STEP 12 — FINAL PIPELINE SUMMARY")

    best_model, worst_model = get_best_worst(results)
    best_row  = comparison_df.iloc[0]
    worst_row = comparison_df.iloc[-1]

    total_time = elapsed(PIPELINE_START)

    print_dict_table(
        {
            "Dataset Shape":        df.shape,
            "Training Samples":     X_train.shape[0],
            "Testing Samples":      X_test.shape[0],
            "Training Features":    X_train.shape[1],
            "---":                  "---",
            "Best Model":           f"{best_model} (Acc={best_row['Accuracy']:.4f})",
            "Worst Model":          f"{worst_model} (Acc={worst_row['Accuracy']:.4f})",
            "Highest Accuracy":     f"{comparison_df['Accuracy'].max():.4f}",
            "Highest Precision":    f"{comparison_df['Precision'].max():.4f}",
            "Highest Recall":       f"{comparison_df['Recall'].max():.4f}",
            "Highest F1 Score":     f"{comparison_df['F1 Score'].max():.4f}",
            "----":                 "----",
            "Total Execution Time": f"{total_time:.2f}s",
            "Saved Models":         MODELS_DIR,
            "Saved Graphs":         GRAPHS_DIR,
            "Saved Results":        RESULTS_DIR,
        },
        title="Summary",
    )


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    # -- Setup ---------------------------------------------------------------
    setup_directories()

    # -- Step 1: Load --------------------------------------------------------
    df_raw = load_data()

    # -- Step 3 (partial): EDA on raw data -----------------------------------
    section_header("STEP 3 — Exploratory Data Analysis")
    plot_missing_value_heatmap(df_raw)
    plot_missing_value_summary(df_raw)
    plot_target_distribution(df_raw)
    plot_age_distribution(df_raw)
    plot_gender_distribution(df_raw)
    plot_race_distribution(df_raw)
    plot_num_diagnoses(df_raw)
    plot_num_medications(df_raw)
    plot_hba1c_distribution(df_raw)
    plot_glucose_distribution(df_raw)
    plot_hospital_stay(df_raw)
    plot_correlation_heatmap(df_raw)

    # -- Step 2: Preprocess --------------------------------------------------
    df_clean, encoders, feature_cols = preprocess_data(df_raw)

    # -- Step 4: Split -------------------------------------------------------
    X_train, X_test, y_train, y_test, scaler = split_data(df_clean, feature_cols)

    # -- Steps 5 & 6: Train + Evaluate ---------------------------------------
    models  = build_models()
    results = train_and_evaluate(
        models, X_train, X_test, y_train, y_test, feature_cols, encoders
    )

    # -- Step 7: Visualizations ----------------------------------------------
    generate_result_visualizations(results, models, feature_cols, encoders)

    # -- Step 8: Comparison table --------------------------------------------
    comparison_df = build_comparison_table(results)

    # -- Step 12: Final summary ----------------------------------------------
    print_final_summary(df_raw, X_train, X_test, results, comparison_df)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
