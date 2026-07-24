"""
preprocessing.py
----------------
Data loading, cleaning, encoding, and splitting for the CDSS
Diabetes Mellitus ML pipeline.
"""

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from utils import (
    DATASET_PATH,
    ENCODERS_PATH,
    FEATURES_PATH,
    SCALER_PATH,
    TARGET_COLUMN,
    RANDOM_STATE,
    TEST_SIZE,
    logger,
    section_header,
    print_dict_table,
)


# =============================================================================
# STEP 1 — Load Data
# =============================================================================
def load_data(path: str = DATASET_PATH) -> pd.DataFrame:
    """
    Load the diabetes dataset from *path* and print basic metadata.

    Parameters
    ----------
    path : str
        Absolute path to the CSV dataset.

    Returns
    -------
    pd.DataFrame
    """
    section_header("STEP 1 — Loading Dataset")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(path, low_memory=False)
    logger.info(f"Dataset loaded from: {path}")

    # ---- Basic metadata ------------------------------------------------
    print_dict_table(
        {
            "Shape":          df.shape,
            "Rows":           df.shape[0],
            "Columns":        df.shape[1],
            "Duplicate Rows": df.duplicated().sum(),
        },
        title="Dataset Overview",
    )

    print("\n  Column Names:")
    for col in df.columns:
        print(f"    - {col}")

    print("\n  Data Types:")
    print(df.dtypes.to_string())

    # ---- Missing values -------------------------------------------------
    missing = df.isnull().sum()
    question_marks = (df == "?").sum()
    combined_missing = missing + question_marks
    print("\n  Missing / '?' Values per Column (top 15):")
    top_missing = combined_missing[combined_missing > 0].sort_values(ascending=False)
    if top_missing.empty:
        print("    None found.")
    else:
        print(top_missing.head(15).to_string())

    print("\n  Basic Statistics:")
    print(df.describe(include="all").T.to_string())

    return df


# =============================================================================
# STEP 2 — Preprocess Data
# =============================================================================
def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, list]:
    """
    Full preprocessing pipeline:
      1. Drop duplicates
      2. Replace '?' with NaN, then impute
      3. Drop columns with excessive missingness
      4. Encode categoricals & target (skipped if dataset is already ML-ready)
      5. Save encoders and feature list

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe from load_data().

    Returns
    -------
    df_clean : pd.DataFrame   — cleaned, encoded dataframe (includes target)
    encoders : dict           — {col_name: LabelEncoder} (may be empty if pre-encoded)
    feature_cols : list[str]  — ordered list of feature column names (no target)
    """
    section_header("STEP 2 — Preprocessing")

    df = df.copy()

    # --- 1. Remove duplicate rows ----------------------------------------
    before = len(df)
    df.drop_duplicates(inplace=True)
    dropped = before - len(df)
    logger.info(f"Removed {dropped} duplicate row(s).")

    # --- 2. Replace '?' with NaN -----------------------------------------
    df.replace("?", np.nan, inplace=True)

    # --- 3. Drop columns with > 40 % missingness -------------------------
    threshold = 0.40
    missing_ratio = df.isnull().mean()
    cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
    if cols_to_drop:
        logger.info(f"Dropping high-missingness columns (>{threshold*100:.0f}%): {cols_to_drop}")
        df.drop(columns=cols_to_drop, inplace=True)

    # --- 4. Drop identifiers that leak or carry no predictive value ------
    id_cols = [c for c in ["encounter_id", "patient_nbr"] if c in df.columns]
    if id_cols:
        df.drop(columns=id_cols, inplace=True)
        logger.info(f"Dropped identifier columns: {id_cols}")

    # --- 5. Separate target ----------------------------------------------
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found in dataset. "
            f"Available columns: {df.columns.tolist()}"
        )

    # --- 6. Impute missing values ----------------------------------------
    for col in df.columns:
        if df[col].dtype == "object":
            mode_val = df[col].mode(dropna=True)
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    logger.info("Missing values imputed.")

    # --- 7. Detect if dataset is already ML-ready (all numeric) ----------
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    encoders: dict = {}

    if not categorical_cols:
        logger.info(
            "Dataset is already fully encoded (no object columns detected). "
            "Skipping LabelEncoder step."
        )
        # Still save an empty encoders dict and feature list for compatibility
        joblib.dump(encoders, ENCODERS_PATH)
        feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
        joblib.dump(feature_cols, FEATURES_PATH)
        logger.info(f"Encoders (empty) saved to: {ENCODERS_PATH}")
        logger.info(f"Feature list saved to: {FEATURES_PATH}")
    else:
        # --- 8. Encode remaining categorical columns ----------------------
        logger.info(f"Encoding {len(categorical_cols)} categorical column(s).")
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

        feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
        joblib.dump(encoders, ENCODERS_PATH)
        joblib.dump(feature_cols, FEATURES_PATH)
        logger.info(f"Encoders saved to: {ENCODERS_PATH}")
        logger.info(f"Feature list saved to: {FEATURES_PATH}")

        # Log target classes if encoded
        target_encoder = encoders.get(TARGET_COLUMN)
        if target_encoder is not None:
            logger.info(f"Target classes: {list(target_encoder.classes_)}")

    # Ensure target is integer
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]

    print_dict_table(
        {
            "Final Shape After Preprocessing": df.shape,
            "Number of Features":              len(feature_cols),
            "Target Column":                   TARGET_COLUMN,
            "Unique Target Classes":           df[TARGET_COLUMN].nunique(),
            "Target Class Distribution":       df[TARGET_COLUMN].value_counts().to_dict(),
        },
        title="Preprocessing Summary",
    )

    return df, encoders, feature_cols



# =============================================================================
# STEP 4 — Split Data
# =============================================================================
def split_data(
    df: pd.DataFrame,
    feature_cols: list,
    scale_features: bool = True,
) -> tuple:
    """
    Split the preprocessed dataframe into stratified train / test sets.
    Optionally apply StandardScaler to the features.

    Parameters
    ----------
    df           : pd.DataFrame — cleaned, encoded dataframe
    feature_cols : list[str]   — feature column names
    scale_features : bool      — if True, fit a StandardScaler on training features

    Returns
    -------
    X_train, X_test, y_train, y_test, scaler (or None)
    """
    section_header("STEP 4 — Splitting Dataset")

    X = df[feature_cols].values
    y = df[TARGET_COLUMN].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)
        joblib.dump(scaler, SCALER_PATH)
        logger.info(f"Scaler saved to: {SCALER_PATH}")

    print_dict_table(
        {
            "Total Samples":  len(df),
            "Training Samples": len(X_train),
            "Testing Samples":  len(X_test),
            "Train %":        f"{100*(1-TEST_SIZE):.0f}%",
            "Test %":         f"{100*TEST_SIZE:.0f}%",
            "Number of Features": X_train.shape[1],
        },
        title="Split Summary",
    )

    return X_train, X_test, y_train, y_test, scaler


# =============================================================================
# REUSABLE PREPROCESSING HELPER (for prediction module)
# =============================================================================
def preprocess_single_patient(patient_dict: dict) -> np.ndarray:
    """
    Preprocess a single patient record for inference.

    Parameters
    ----------
    patient_dict : dict
        Raw patient data as key-value pairs (column_name → value).

    Returns
    -------
    np.ndarray of shape (1, n_features)
    """
    encoders: dict    = joblib.load(ENCODERS_PATH)
    feature_cols: list = joblib.load(FEATURES_PATH)

    # Build a dataframe row
    row = pd.DataFrame([patient_dict])
    row.replace("?", np.nan, inplace=True)

    # Impute missing
    for col in row.columns:
        if row[col].isnull().any():
            row[col] = row[col].fillna("Unknown" if row[col].dtype == "object" else 0)

    # Encode categoricals using saved encoders
    for col in row.select_dtypes(include=["object"]).columns:
        if col in encoders:
            le = encoders[col]
            try:
                row[col] = le.transform(row[col].astype(str))
            except ValueError:
                # Unseen label — map to most frequent class (index 0)
                row[col] = 0
        else:
            row[col] = 0

    # Ensure all expected feature columns are present (fill with 0 if missing)
    for fc in feature_cols:
        if fc not in row.columns:
            row[fc] = 0

    return row[feature_cols].values.astype(float)
