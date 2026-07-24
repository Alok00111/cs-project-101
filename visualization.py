"""
visualization.py
----------------
All EDA and result visualization functions.
Uses ONLY matplotlib (no seaborn).
All plots saved as high-resolution PNG to graphs/.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from utils import GRAPHS_DIR, TARGET_COLUMN, logger

# Use non-interactive backend (safe for scripts with no display)
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# GLOBAL STYLE
# ---------------------------------------------------------------------------
PALETTE = [
    "#2563EB",   # blue
    "#16A34A",   # green
    "#DC2626",   # red
    "#D97706",   # amber
    "#7C3AED",   # violet
    "#0891B2",   # cyan
    "#DB2777",   # pink
    "#65A30D",   # lime
]

plt.rcParams.update(
    {
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.edgecolor":    "#CCCCCC",
        "axes.labelcolor":   "#1F2937",
        "axes.titlecolor":   "#1F2937",
        "xtick.color":       "#4B5563",
        "ytick.color":       "#4B5563",
        "axes.grid":         True,
        "grid.color":        "#E5E7EB",
        "grid.linestyle":    "--",
        "grid.linewidth":    0.6,
        "font.family":       "DejaVu Sans",
        "font.size":         11,
        "axes.titlesize":    13,
        "axes.labelsize":    11,
        "legend.fontsize":   10,
    }
)

DPI = 300


def _save(fig: plt.Figure, filename: str) -> str:
    """Save *fig* to graphs/ and return the full path."""
    path = os.path.join(GRAPHS_DIR, filename)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {path}")
    return path


# =============================================================================
# EDA — Dataset Overview
# =============================================================================
def plot_missing_value_heatmap(df: pd.DataFrame) -> str:
    """
    Heatmap of missing / NaN values across columns (matplotlib imshow).
    """
    import numpy as np

    # Replace '?' already done at this point; show NaN
    missing_mask = df.isnull().astype(int)

    # Limit to columns with at least one missing value
    cols_with_missing = missing_mask.columns[missing_mask.sum() > 0]
    if cols_with_missing.empty:
        logger.info("No missing values to plot.")
        return ""

    mask_subset = missing_mask[cols_with_missing]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(mask_subset.T.values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_yticks(range(len(cols_with_missing)))
    ax.set_yticklabels(cols_with_missing, fontsize=8)
    ax.set_xlabel("Patient Records (row index)")
    ax.set_ylabel("Feature")
    ax.set_title("Missing Value Heatmap")
    fig.tight_layout()
    return _save(fig, "01_missing_value_heatmap.png")


def plot_missing_value_summary(df: pd.DataFrame) -> str:
    """Bar chart: count of missing values per column."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        logger.info("No missing values to summarize.")
        return ""

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(missing)), missing.values, color=PALETTE[0], edgecolor="white")
    ax.set_xticks(range(len(missing)))
    ax.set_xticklabels(missing.index, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Missing Count")
    ax.set_title("Missing Values Summary per Column")
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(int(bar.get_height())),
            ha="center", va="bottom", fontsize=8,
        )
    fig.tight_layout()
    return _save(fig, "02_missing_value_summary.png")


# =============================================================================
# EDA — Target Variable
# =============================================================================
def plot_target_distribution(df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> str:
    """Bar chart of the readmission target class distribution."""
    counts = df[target_col].value_counts().sort_index()
    labels = counts.index.astype(str)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, counts.values, color=PALETTE[:len(labels)], edgecolor="white", width=0.5)
    ax.set_xlabel("Readmission Class")
    ax.set_ylabel("Number of Patients")
    ax.set_title("Readmission Class Distribution (Target Variable)")

    total = counts.sum()
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.002,
            f"{val:,}\n({100*val/total:.1f}%)",
            ha="center", va="bottom", fontsize=9,
        )
    fig.tight_layout()
    return _save(fig, "03_target_distribution.png")


# =============================================================================
# EDA — Demographics
# =============================================================================
def plot_age_distribution(df: pd.DataFrame) -> str:
    """Bar / histogram of age groups."""
    age_col = _find_col(df, ["age", "Age"])
    if age_col is None:
        logger.warning("Age column not found — skipping.")
        return ""

    fig, ax = plt.subplots(figsize=(10, 5))
    series = df[age_col]
    if series.dtype == "object":
        counts = series.value_counts().sort_index()
        ax.bar(counts.index.astype(str), counts.values, color=PALETTE[1], edgecolor="white")
        ax.set_xticklabels(counts.index.astype(str), rotation=45, ha="right")
    else:
        ax.hist(series, bins=20, color=PALETTE[1], edgecolor="white")

    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    ax.set_title("Patient Age Distribution")
    fig.tight_layout()
    return _save(fig, "04_age_distribution.png")


def plot_gender_distribution(df: pd.DataFrame) -> str:
    gender_col = _find_col(df, ["gender", "Gender", "sex", "Sex"])
    if gender_col is None:
        logger.warning("Gender column not found — skipping.")
        return ""

    counts = df[gender_col].value_counts()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(counts.index.astype(str), counts.values, color=PALETTE[0:len(counts)], edgecolor="white", width=0.4)
    ax.set_xlabel("Gender")
    ax.set_ylabel("Count")
    ax.set_title("Gender Distribution")
    for i, (idx, val) in enumerate(counts.items()):
        ax.text(i, val + len(df) * 0.003, f"{val:,}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    return _save(fig, "05_gender_distribution.png")


def plot_race_distribution(df: pd.DataFrame) -> str:
    race_col = _find_col(df, ["race", "Race", "ethnicity"])
    if race_col is None:
        logger.warning("Race column not found — skipping.")
        return ""

    counts = df[race_col].value_counts()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(counts.index.astype(str), counts.values,
           color=PALETTE[:len(counts)], edgecolor="white", width=0.5)
    ax.set_xticklabels(counts.index.astype(str), rotation=30, ha="right")
    ax.set_xlabel("Race / Ethnicity")
    ax.set_ylabel("Count")
    ax.set_title("Race / Ethnicity Distribution")
    fig.tight_layout()
    return _save(fig, "06_race_distribution.png")


# =============================================================================
# EDA — Clinical Information
# =============================================================================
def plot_num_diagnoses(df: pd.DataFrame) -> str:
    col = _find_col(df, ["number_diagnoses", "num_diagnoses", "num_diagnosis"])
    return _plot_numeric_hist(df, col, "Number of Diagnoses", "07_num_diagnoses.png", color=PALETTE[4])


def plot_num_medications(df: pd.DataFrame) -> str:
    col = _find_col(df, ["num_medications", "number_medications", "medications"])
    return _plot_numeric_hist(df, col, "Number of Medications", "08_num_medications.png", color=PALETTE[5])


def plot_hba1c_distribution(df: pd.DataFrame) -> str:
    col = _find_col(df, ["A1Cresult", "a1cresult", "hba1c", "HbA1c", "A1C_result"])
    if col is None:
        logger.warning("HbA1c column not found — skipping.")
        return ""
    if df[col].dtype == "object":
        counts = df[col].value_counts()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(counts.index.astype(str), counts.values, color=PALETTE[2], edgecolor="white")
        ax.set_xticklabels(counts.index.astype(str), rotation=30, ha="right")
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df[col].dropna(), bins=20, color=PALETTE[2], edgecolor="white")
    ax.set_xlabel("HbA1c Category")
    ax.set_ylabel("Count")
    ax.set_title("HbA1c Distribution")
    fig.tight_layout()
    return _save(fig, "09_hba1c_distribution.png")


def plot_glucose_distribution(df: pd.DataFrame) -> str:
    col = _find_col(df, ["max_glu_serum", "glucose", "blood_glucose", "gluco"])
    if col is None:
        logger.warning("Glucose column not found — skipping.")
        return ""
    if df[col].dtype == "object":
        counts = df[col].value_counts()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(counts.index.astype(str), counts.values, color=PALETTE[3], edgecolor="white")
        ax.set_xticklabels(counts.index.astype(str), rotation=30, ha="right")
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df[col].dropna(), bins=20, color=PALETTE[3], edgecolor="white")
    ax.set_xlabel("Blood Glucose Serum")
    ax.set_ylabel("Count")
    ax.set_title("Blood Glucose Serum Distribution")
    fig.tight_layout()
    return _save(fig, "10_glucose_distribution.png")


def plot_hospital_stay(df: pd.DataFrame) -> str:
    col = _find_col(df, ["time_in_hospital", "hospital_stay", "days_in_hospital"])
    return _plot_numeric_hist(df, col, "Hospital Stay (Days)", "11_hospital_stay.png", color=PALETTE[6])


# =============================================================================
# EDA — Correlation Heatmap
# =============================================================================
def plot_correlation_heatmap(df: pd.DataFrame, max_cols: int = 25) -> str:
    """
    Correlation heatmap using matplotlib matshow.
    Uses only numeric columns (up to *max_cols* highest-variance ones).
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] > max_cols:
        # Pick highest-variance columns
        var_sorted = numeric_df.var().sort_values(ascending=False)
        numeric_df = numeric_df[var_sorted.index[:max_cols]]

    corr = numeric_df.corr()
    n = corr.shape[0]

    fig, ax = plt.subplots(figsize=(max(10, n // 2), max(8, n // 2)))
    im = ax.matshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.columns, fontsize=7)
    ax.set_title("Correlation Heatmap (Numeric Features)", pad=20)
    fig.tight_layout()
    return _save(fig, "12_correlation_heatmap.png")


# =============================================================================
# RESULTS — Model Comparison Bar Charts
# =============================================================================
def plot_metric_comparison(
    results: dict,
    metric: str,
    filename: str,
    title: str,
    ylabel: str,
) -> str:
    """
    Generic bar chart comparing a single metric across all models.

    Parameters
    ----------
    results : dict   {model_name: {metric: value, ...}}
    metric  : str    key inside each inner dict
    """
    names  = list(results.keys())
    values = [results[n].get(metric, 0) for n in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, values, color=PALETTE[:len(names)], edgecolor="white", width=0.5)
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, min(1.1, max(values) * 1.15 + 0.01))
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=9,
        )
    fig.tight_layout()
    return _save(fig, filename)


def plot_all_metric_comparisons(results: dict) -> list[str]:
    """Generate bar charts for Accuracy, Precision, Recall, F1."""
    saved = []
    metrics = [
        ("accuracy",  "13_accuracy_comparison.png",  "Accuracy Comparison",  "Accuracy"),
        ("precision", "14_precision_comparison.png", "Precision Comparison", "Precision (Weighted)"),
        ("recall",    "15_recall_comparison.png",    "Recall Comparison",    "Recall (Weighted)"),
        ("f1_score",  "16_f1_comparison.png",        "F1 Score Comparison",  "F1 Score (Weighted)"),
    ]
    for metric, fname, title, ylabel in metrics:
        p = plot_metric_comparison(results, metric, fname, title, ylabel)
        saved.append(p)
    return saved


def plot_time_comparisons(results: dict) -> list[str]:
    """Bar charts for Training Time and Prediction Time."""
    saved = []
    time_metrics = [
        ("train_time", "17_training_time.png", "Training Time Comparison", "Time (seconds)"),
        ("pred_time",  "18_prediction_time.png","Prediction Time Comparison","Time (seconds)"),
    ]
    names = list(results.keys())
    for metric, fname, title, ylabel in time_metrics:
        values = [results[n].get(metric, 0) for n in names]
        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(names, values, color=PALETTE[:len(names)], edgecolor="white", width=0.5)
        ax.set_xlabel("Model")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                f"{val:.4f}s",
                ha="center", va="bottom", fontsize=9,
            )
        fig.tight_layout()
        saved.append(_save(fig, fname))
    return saved


# =============================================================================
# RESULTS — ROC Curves
# =============================================================================
def plot_roc_curves(results: dict) -> str:
    """
    Multi-class ROC curves (one-vs-rest) for all models.
    Each model's macro-average ROC curve is drawn on a single plot.
    """
    from sklearn.metrics import roc_curve, auc

    fig, ax = plt.subplots(figsize=(9, 7))

    for i, (name, info) in enumerate(results.items()):
        fpr  = info.get("fpr_macro")
        tpr  = info.get("tpr_macro")
        roc_auc = info.get("roc_auc", 0)
        if fpr is not None and tpr is not None:
            ax.plot(fpr, tpr, color=PALETTE[i % len(PALETTE)],
                    lw=2, label=f"{name} (AUC = {roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Classifier")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models (Macro Average, One-vs-Rest)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return _save(fig, "19_roc_curves.png")


# =============================================================================
# RESULTS — Confusion Matrices
# =============================================================================
def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    model_name: str,
    file_index: int,
) -> str:
    """Plot a single confusion matrix."""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix — {model_name}")

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]:,}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "#1F2937",
                fontsize=10,
            )
    fig.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    return _save(fig, f"{20 + file_index}_cm_{safe_name}.png")


def plot_all_confusion_matrices(results: dict) -> list[str]:
    saved = []
    for i, (name, info) in enumerate(results.items()):
        cm           = info.get("confusion_matrix")
        class_names  = info.get("class_names", [str(c) for c in range(3)])
        if cm is not None:
            saved.append(plot_confusion_matrix(cm, class_names, name, i))
    return saved


# =============================================================================
# RESULTS — Decision Tree Visualization
# =============================================================================
def plot_decision_tree(model, feature_names: list, class_names: list) -> str:
    """Export a text-level visualization of the Decision Tree."""
    from sklearn.tree import export_text, plot_tree

    fig, ax = plt.subplots(figsize=(22, 10))
    plot_tree(
        model,
        max_depth=4,
        feature_names=feature_names,
        class_names=[str(c) for c in class_names],
        filled=True,
        ax=ax,
        fontsize=8,
        impurity=False,
        proportion=False,
    )
    ax.set_title("Decision Tree Visualization (Max Depth = 4)")
    fig.tight_layout()
    return _save(fig, "24_decision_tree.png")


# =============================================================================
# RESULTS — Feature Importance
# =============================================================================
def plot_feature_importance(
    importances: np.ndarray,
    feature_names: list,
    model_name: str,
    file_prefix: str,
    top_n: int = 20,
) -> str:
    """Horizontal bar chart of top-N feature importances."""
    indices = np.argsort(importances)[::-1][:top_n]
    top_feats = [feature_names[i] for i in indices]
    top_vals  = importances[indices]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n // 2)))
    ax.barh(range(top_n), top_vals[::-1], color=PALETTE[0], edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_feats[::-1], fontsize=9)
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}")
    fig.tight_layout()
    return _save(fig, f"{file_prefix}_feature_importance.png")


# =============================================================================
# INTERNAL HELPERS
# =============================================================================
def _find_col(df: pd.DataFrame, candidates: list) -> str | None:
    """Return the first candidate column name that exists in *df*, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    # Case-insensitive fallback
    lower_map = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def _plot_numeric_hist(
    df: pd.DataFrame,
    col: str | None,
    label: str,
    filename: str,
    color: str = "#2563EB",
) -> str:
    if col is None:
        logger.warning(f"Column for '{label}' not found — skipping.")
        return ""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df[col].dropna(), bins=25, color=color, edgecolor="white")
    ax.set_xlabel(label)
    ax.set_ylabel("Count")
    ax.set_title(f"{label} Distribution")
    fig.tight_layout()
    return _save(fig, filename)
