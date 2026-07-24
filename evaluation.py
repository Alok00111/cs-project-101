"""
evaluation.py
-------------
Model evaluation, metrics computation, and comparison table generation.
"""

import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

from utils import COMPARISON_CSV_PATH, logger, section_header, print_dict_table


# =============================================================================
# PER-MODEL EVALUATION
# =============================================================================
def evaluate_model(
    name: str,
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_time: float,
    pred_time: float,
    class_names: list | None = None,
) -> dict:
    """
    Compute all evaluation metrics for a single trained model.

    Parameters
    ----------
    name       : Human-readable model name.
    model      : Fitted sklearn-compatible estimator.
    X_test     : Test features.
    y_test     : True test labels.
    X_train    : Training features (for computing train accuracy).
    y_train    : True training labels.
    train_time : Seconds taken to train.
    pred_time  : Seconds taken to predict on test set.
    class_names: Optional list of class label strings.

    Returns
    -------
    dict with all metrics plus raw objects (cm, fpr, tpr, etc.)
    """
    y_pred = model.predict(X_test)

    # ---- Probabilities for ROC AUC ----------------------------------------
    n_classes = len(np.unique(y_test))
    classes   = np.unique(y_test)

    try:
        y_proba = model.predict_proba(X_test)
    except AttributeError:
        y_proba = None

    # ---- Core metrics -------------------------------------------------------
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # ---- ROC AUC (macro, one-vs-rest) --------------------------------------
    roc_auc = 0.0
    fpr_macro, tpr_macro = None, None

    if y_proba is not None and n_classes >= 2:
        try:
            if n_classes == 2:
                roc_auc = roc_auc_score(y_test, y_proba[:, 1])
                fpr_macro, tpr_macro, _ = roc_curve(y_test, y_proba[:, 1])
            else:
                y_bin = label_binarize(y_test, classes=classes)
                roc_auc = roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro")
                # Macro-average ROC curve
                all_fpr = np.unique(np.concatenate(
                    [roc_curve(y_bin[:, i], y_proba[:, i])[0] for i in range(n_classes)]
                ))
                mean_tpr = np.zeros_like(all_fpr)
                for i in range(n_classes):
                    f, t, _ = roc_curve(y_bin[:, i], y_proba[:, i])
                    mean_tpr += np.interp(all_fpr, f, t)
                mean_tpr /= n_classes
                fpr_macro, tpr_macro = all_fpr, mean_tpr
        except Exception as exc:
            logger.warning(f"ROC AUC computation failed for {name}: {exc}")

    # ---- Confusion matrix --------------------------------------------------
    cm = confusion_matrix(y_test, y_pred)

    # ---- Class names -------------------------------------------------------
    if class_names is None:
        class_names = [str(c) for c in classes]

    # ---- Report string -----------------------------------------------------
    report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)

    # ---- Train accuracy (sanity check) -------------------------------------
    train_acc = accuracy_score(y_train, model.predict(X_train))

    result = {
        "name":             name,
        "accuracy":         round(acc, 6),
        "precision":        round(prec, 6),
        "recall":           round(rec, 6),
        "f1_score":         round(f1, 6),
        "roc_auc":          round(roc_auc, 6),
        "train_accuracy":   round(train_acc, 6),
        "train_time":       train_time,
        "pred_time":        pred_time,
        "confusion_matrix": cm,
        "class_names":      class_names,
        "fpr_macro":        fpr_macro,
        "tpr_macro":        tpr_macro,
        "y_pred":           y_pred,
        "y_proba":          y_proba,
        "report":           report,
    }

    _print_metrics(result)
    return result


def _print_metrics(r: dict) -> None:
    section_header(f"Results — {r['name']}")
    print_dict_table(
        {
            "Accuracy":       r["accuracy"],
            "Precision":      r["precision"],
            "Recall":         r["recall"],
            "F1 Score":       r["f1_score"],
            "ROC AUC":        r["roc_auc"],
            "Train Accuracy": r["train_accuracy"],
            "Training Time":  f"{r['train_time']:.4f}s",
            "Prediction Time":f"{r['pred_time']:.4f}s",
        }
    )
    print("\n  Classification Report:")
    print(r["report"])


# =============================================================================
# MODEL COMPARISON TABLE
# =============================================================================
def build_comparison_table(results: dict) -> pd.DataFrame:
    """
    Build a sorted comparison DataFrame and export to CSV.

    Parameters
    ----------
    results : dict   {model_name: metrics_dict}

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    for name, r in results.items():
        rows.append(
            {
                "Model":           name,
                "Accuracy":        r["accuracy"],
                "Precision":       r["precision"],
                "Recall":          r["recall"],
                "F1 Score":        r["f1_score"],
                "ROC AUC":         r["roc_auc"],
                "Training Time(s)":r["train_time"],
                "Prediction Time(s)": r["pred_time"],
            }
        )

    df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    df.to_csv(COMPARISON_CSV_PATH, index=False)
    logger.info(f"Comparison table saved to: {COMPARISON_CSV_PATH}")

    section_header("MODEL COMPARISON TABLE")
    print(df.to_string(index=False))
    return df


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================
def get_best_worst(results: dict) -> tuple[str, str]:
    """Return (best_model_name, worst_model_name) by accuracy."""
    sorted_by_acc = sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    return sorted_by_acc[0][0], sorted_by_acc[-1][0]
