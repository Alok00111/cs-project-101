"""
utils.py
--------
Utility constants, path configuration, and helper functions for the CDSS
Diabetes Mellitus ML pipeline.
"""

import os
import time
import logging

# =============================================================================
# DATASET PATHS  — update these paths if your dataset location changes
# =============================================================================
DATASET_PATH = r"C:\Users\aloks\Downloads\final_ml_ready_dataset.csv"
DRUG_KNOWLEDGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "drug_knowledge.csv")

# =============================================================================
# OUTPUT DIRECTORIES
# =============================================================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
GRAPHS_DIR = os.path.join(BASE_DIR, "graphs")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# =============================================================================
# MODEL SAVE PATHS
# =============================================================================
LR_MODEL_PATH      = os.path.join(MODELS_DIR, "logistic_regression.pkl")
DT_MODEL_PATH      = os.path.join(MODELS_DIR, "decision_tree.pkl")
RF_MODEL_PATH      = os.path.join(MODELS_DIR, "random_forest.pkl")
XGB_MODEL_PATH     = os.path.join(MODELS_DIR, "xgboost.pkl")
ENCODERS_PATH      = os.path.join(MODELS_DIR, "label_encoders.pkl")
FEATURES_PATH      = os.path.join(MODELS_DIR, "feature_columns.pkl")
SCALER_PATH        = os.path.join(MODELS_DIR, "scaler.pkl")

# Results
COMPARISON_CSV_PATH = os.path.join(RESULTS_DIR, "model_comparison.csv")

# =============================================================================
# RANDOM STATE
# =============================================================================
RANDOM_STATE = 42
TEST_SIZE    = 0.30

# =============================================================================
# TARGET VARIABLE
# =============================================================================
TARGET_COLUMN = "readmitted"

# =============================================================================
# LOGGING SETUP
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# DIRECTORY SETUP
# =============================================================================
def setup_directories() -> None:
    """Create output directories if they do not already exist."""
    for directory in [GRAPHS_DIR, MODELS_DIR, RESULTS_DIR]:
        os.makedirs(directory, exist_ok=True)
    logger.info("Output directories are ready.")


# =============================================================================
# TIMING HELPERS
# =============================================================================
def start_timer() -> float:
    """Return the current high-resolution time."""
    return time.perf_counter()


def elapsed(start: float) -> float:
    """Return elapsed seconds since *start* (2 decimal places)."""
    return round(time.perf_counter() - start, 4)


# =============================================================================
# PRETTY PRINTING
# =============================================================================
def section_header(title: str) -> None:
    """Print a formatted section header to stdout."""
    width = 70
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_dict_table(data: dict, title: str = "") -> None:
    """Print a dictionary as a two-column aligned table."""
    if title:
        print(f"\n{title}")
        print("-" * 50)
    for key, value in data.items():
        print(f"  {str(key):<35} {str(value)}")
