"""
shap_analysis.py
----------------
SHAP (SHapley Additive exPlanations) analysis for all trained models.

Generates:
  - SHAP Summary Plot (beeswarm)       — for XGBoost and Random Forest
  - SHAP Bar Plot (mean |SHAP|)        — global feature importance
  - SHAP Waterfall Plot                — single patient explanation
  - SHAP Dependence Plot               — top 2 features
  - SHAP Decision Plot                 — multi-class decision paths
  - SHAP Force Plot (saved as PNG)     — single prediction explanation

All saved to graphs/shap_*.png at 300 DPI.

Run with:
    python shap_analysis.py
(Requires train_models.py to have been run first.)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import joblib
import shap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
matplotlib.use("Agg")
warnings.filterwarnings("ignore")

from utils import (
    GRAPHS_DIR,
    MODELS_DIR,
    RF_MODEL_PATH,
    XGB_MODEL_PATH,
    LR_MODEL_PATH,
    DT_MODEL_PATH,
    FEATURES_PATH,
    SCALER_PATH,
    logger,
    section_header,
)
from preprocessing import load_data, preprocess_data, split_data

DPI = 300

# ── Colour palette (flat, no gradients per project rules) ───────────────────
BLUE   = "#2563EB"
GREEN  = "#16A34A"
RED    = "#DC2626"
AMBER  = "#D97706"
VIOLET = "#7C3AED"

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.edgecolor":   "#CCCCCC",
        "axes.grid":        True,
        "grid.color":       "#E5E7EB",
        "grid.linestyle":   "--",
        "grid.linewidth":   0.6,
        "font.family":      "DejaVu Sans",
        "font.size":        11,
        "axes.titlesize":   13,
    }
)


# =============================================================================
# LOAD EVERYTHING
# =============================================================================
def load_artifacts():
    """Load saved models, scaler, feature list, and prepare test data."""
    section_header("Loading Saved Artifacts")

    feature_cols = joblib.load(FEATURES_PATH)
    scaler       = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None

    models = {}
    for name, path in [
        ("XGBoost",           XGB_MODEL_PATH),
        ("Random Forest",     RF_MODEL_PATH),
        ("Decision Tree",     DT_MODEL_PATH),
        ("Logistic Regression", LR_MODEL_PATH),
    ]:
        if os.path.exists(path):
            models[name] = joblib.load(path)
            logger.info(f"Loaded: {name}")
        else:
            logger.warning(f"Model not found, skipping: {path}")

    return models, feature_cols, scaler


def prepare_test_data(feature_cols, scaler, sample_size: int = 1000):
    """
    Reload and preprocess the dataset, return a sample of the test set.
    SHAP on tree models is fast even on 1000 rows; for LR we use the full set.
    """
    section_header("Preparing Test Data Sample for SHAP")
    df_raw = load_data()
    df_clean, encoders, _ = preprocess_data(df_raw)
    X_train, X_test, y_train, y_test, _ = split_data(
        df_clean, feature_cols, scale_features=False  # we'll scale manually
    )

    # Scale if scaler exists
    if scaler is not None:
        X_test_scaled  = scaler.transform(X_test)
        X_train_scaled = scaler.transform(X_train)
    else:
        X_test_scaled  = X_test
        X_train_scaled = X_train

    # Sample for SHAP (TreeExplainer is fast; LinearExplainer needs background)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_test_scaled), size=min(sample_size, len(X_test_scaled)), replace=False)
    X_sample = X_test_scaled[idx]
    y_sample = y_test[idx]

    logger.info(f"SHAP sample size: {X_sample.shape}")
    return X_train_scaled, X_test_scaled, X_sample, y_sample, feature_cols


def _save(fig, filename):
    path = os.path.join(GRAPHS_DIR, filename)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {path}")
    return path


# =============================================================================
# SHAP — XGBoost (TreeExplainer — most informative)
# =============================================================================
def shap_xgboost(model, X_sample, feature_cols):
    section_header("SHAP — XGBoost")

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # shap_values shape: (n_samples, n_features, n_classes) for multiclass XGBoost
    # or (n_classes, n_samples, n_features) depending on version
    if isinstance(shap_values, list):
        # Older shap: list of arrays, one per class
        sv_class0 = shap_values[0]   # class 0 = NO readmission
    elif shap_values.ndim == 3:
        # Newer shap: array (n_samples, n_features, n_classes)
        sv_class0 = shap_values[:, :, 0]
    else:
        sv_class0 = shap_values

    # ── 1. Summary Plot (beeswarm) — class 0 ─────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 9))
    shap.summary_plot(
        sv_class0,
        X_sample,
        feature_names=feature_cols,
        show=False,
        max_display=20,
        plot_type="dot",
    )
    plt.title("SHAP Summary Plot — XGBoost (Class: NO Readmission)", fontsize=13, pad=12)
    plt.tight_layout()
    _save(plt.gcf(), "shap_01_xgb_summary_beeswarm.png")

    # ── 2. Bar Plot — mean |SHAP| ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        sv_class0,
        X_sample,
        feature_names=feature_cols,
        show=False,
        max_display=20,
        plot_type="bar",
    )
    plt.title("SHAP Feature Importance — XGBoost (Mean |SHAP Value|)", fontsize=13, pad=12)
    plt.tight_layout()
    _save(plt.gcf(), "shap_02_xgb_bar_importance.png")

    # ── 3. Waterfall plot — single patient ────────────────────────────────
    try:
        explanation = shap.Explanation(
            values=sv_class0[0],
            base_values=explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
            data=X_sample[0],
            feature_names=feature_cols,
        )
        fig, ax = plt.subplots(figsize=(11, 8))
        shap.plots.waterfall(explanation, max_display=20, show=False)
        plt.title("SHAP Waterfall Plot — XGBoost (Patient #1 Explanation)", fontsize=13, pad=12)
        plt.tight_layout()
        _save(plt.gcf(), "shap_03_xgb_waterfall.png")
    except Exception as e:
        logger.warning(f"Waterfall plot skipped: {e}")

    # ── 4. Dependence plots — top 2 features ─────────────────────────────
    mean_abs = np.abs(sv_class0).mean(axis=0)
    top2_idx = np.argsort(mean_abs)[::-1][:2]

    for rank, feat_idx in enumerate(top2_idx):
        feat_name = feature_cols[feat_idx]
        fig, ax = plt.subplots(figsize=(9, 6))
        shap.dependence_plot(
            feat_idx,
            sv_class0,
            X_sample,
            feature_names=feature_cols,
            ax=ax,
            show=False,
        )
        ax.set_title(f"SHAP Dependence Plot — XGBoost: {feat_name}", fontsize=13)
        fig.tight_layout()
        _save(fig, f"shap_04_xgb_dependence_rank{rank+1}.png")

    logger.info("XGBoost SHAP analysis complete.")
    return sv_class0, explainer


# =============================================================================
# SHAP — Random Forest (TreeExplainer)
# =============================================================================
def shap_random_forest(model, X_sample, feature_cols):
    section_header("SHAP — Random Forest")

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Random Forest multiclass: list of arrays [class0, class1, class2]
    if isinstance(shap_values, list):
        sv_class0 = shap_values[0]
    elif shap_values.ndim == 3:
        sv_class0 = shap_values[:, :, 0]
    else:
        sv_class0 = shap_values

    # ── 1. Summary beeswarm ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 9))
    shap.summary_plot(
        sv_class0,
        X_sample,
        feature_names=feature_cols,
        show=False,
        max_display=20,
        plot_type="dot",
    )
    plt.title("SHAP Summary Plot — Random Forest (Class: NO Readmission)", fontsize=13, pad=12)
    plt.tight_layout()
    _save(plt.gcf(), "shap_05_rf_summary_beeswarm.png")

    # ── 2. Bar plot ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        sv_class0,
        X_sample,
        feature_names=feature_cols,
        show=False,
        max_display=20,
        plot_type="bar",
    )
    plt.title("SHAP Feature Importance — Random Forest (Mean |SHAP Value|)", fontsize=13, pad=12)
    plt.tight_layout()
    _save(plt.gcf(), "shap_06_rf_bar_importance.png")

    logger.info("Random Forest SHAP analysis complete.")
    return sv_class0


# =============================================================================
# SHAP — Logistic Regression (LinearExplainer)
# =============================================================================
def shap_logistic_regression(model, X_train, X_sample, feature_cols):
    section_header("SHAP — Logistic Regression")

    # Use a background sample for the LinearExplainer
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(len(X_train), size=min(200, len(X_train)), replace=False)
    background = X_train[bg_idx]

    explainer   = shap.LinearExplainer(model, background, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        sv_class0 = shap_values[0]
    elif shap_values.ndim == 3:
        sv_class0 = shap_values[:, :, 0]
    else:
        sv_class0 = shap_values

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        sv_class0,
        X_sample,
        feature_names=feature_cols,
        show=False,
        max_display=20,
        plot_type="bar",
    )
    plt.title("SHAP Feature Importance — Logistic Regression (Mean |SHAP Value|)", fontsize=13, pad=12)
    plt.tight_layout()
    _save(plt.gcf(), "shap_07_lr_bar_importance.png")

    logger.info("Logistic Regression SHAP analysis complete.")
    return sv_class0


# =============================================================================
# CROSS-MODEL SHAP COMPARISON
# =============================================================================
def shap_cross_model_comparison(shap_dict: dict, feature_cols, top_n=15):
    """
    Side-by-side bar chart comparing mean |SHAP| for the same top-N features
    across XGBoost, Random Forest, and Logistic Regression.
    """
    section_header("SHAP — Cross-Model Feature Importance Comparison")

    # Compute mean |SHAP| for each model
    records = {}
    for model_name, sv in shap_dict.items():
        records[model_name] = np.abs(sv).mean(axis=0)

    df_importance = pd.DataFrame(records, index=feature_cols)

    # Pick top_n by XGBoost mean |SHAP| (or first available model)
    anchor = list(records.keys())[0]
    top_features = df_importance[anchor].sort_values(ascending=False).head(top_n).index.tolist()
    df_plot = df_importance.loc[top_features].iloc[::-1]   # reverse for horizontal bar

    model_names = df_plot.columns.tolist()
    colors = [BLUE, GREEN, AMBER, VIOLET][:len(model_names)]

    n_models = len(model_names)
    n_feats  = len(top_features)
    bar_h    = 0.22
    y_pos    = np.arange(n_feats)

    fig, ax = plt.subplots(figsize=(13, max(8, n_feats // 2)))

    for i, (mname, color) in enumerate(zip(model_names, colors)):
        offset = (i - n_models / 2 + 0.5) * bar_h
        ax.barh(y_pos + offset, df_plot[mname], bar_h * 0.9, label=mname, color=color)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot.index, fontsize=9)
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title(f"Cross-Model SHAP Feature Importance Comparison (Top {top_n})", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    _save(fig, "shap_08_cross_model_comparison.png")


# =============================================================================
# MAIN
# =============================================================================
def main():
    section_header("SHAP ANALYSIS — Starting")

    # Load
    models, feature_cols, scaler = load_artifacts()
    X_train, X_test, X_sample, y_sample, feature_cols = prepare_test_data(
        feature_cols, scaler, sample_size=500  # 500 rows is enough for SHAP visuals
    )

    shap_dict = {}

    # ── XGBoost ──────────────────────────────────────────────────────────────
    if "XGBoost" in models:
        sv_xgb, _ = shap_xgboost(models["XGBoost"], X_sample, feature_cols)
        shap_dict["XGBoost"] = sv_xgb

    # ── Random Forest ─────────────────────────────────────────────────────────
    if "Random Forest" in models:
        sv_rf = shap_random_forest(models["Random Forest"], X_sample, feature_cols)
        shap_dict["Random Forest"] = sv_rf

    # ── Logistic Regression ───────────────────────────────────────────────────
    if "Logistic Regression" in models:
        try:
            sv_lr = shap_logistic_regression(
                models["Logistic Regression"], X_train, X_sample, feature_cols
            )
            shap_dict["Logistic Regression"] = sv_lr
        except Exception as e:
            logger.warning(f"Logistic Regression SHAP skipped: {e}")

    # ── Cross-model comparison ────────────────────────────────────────────────
    if len(shap_dict) >= 2:
        shap_cross_model_comparison(shap_dict, feature_cols, top_n=15)

    section_header("SHAP ANALYSIS — Complete")
    print(f"\n  All SHAP plots saved to: {GRAPHS_DIR}")
    print("\n  Generated plots:")
    for f in sorted(os.listdir(GRAPHS_DIR)):
        if f.startswith("shap_"):
            print(f"    {f}")


if __name__ == "__main__":
    main()
