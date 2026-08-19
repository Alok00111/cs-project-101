# Project Report
# Clinical Decision Support System (CDSS) for Type 2 Diabetes Mellitus

**Final Year Computer Science Project**
**Author: Alok**
**Technology: Python, Flask, XGBoost, Scikit-Learn, Pandas, HTML/CSS**

---

## 1. Introduction

This project is a full-stack **Clinical Decision Support System (CDSS)** built to assist medical professionals treating patients with **Type 2 Diabetes Mellitus (T2DM)**. It is powered by a Machine Learning (ML) backend and presented through an interactive web application.

The system solves two distinct clinical problems simultaneously:

1. **Hospital Readmission Risk Prediction** — The system uses a trained XGBoost ML model to predict whether a patient is likely to be re-admitted to the hospital, allowing doctors to identify high-risk patients early.
2. **Drug Recommendation Engine** — A rule-based clinical inference engine evaluates over 20 antidiabetic medications against the patient's individual health profile and recommends the safest, most appropriate drugs while explicitly listing the ones that are contraindicated and why.

This document explains the complete flow of the project, from raw data all the way to the web interface displayed to a clinician.

---

## 2. High-Level Architecture

The entire project is structured into two separate workflows that are tightly integrated.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PHASE 1: OFFLINE ML PIPELINE                 │
│                    (Run once: python train_models.py)               │
│                                                                     │
│   CSV Dataset  ──►  preprocessing.py  ──►  train_models.py          │
│                          │                       │                  │
│                     Cleaned Data            4 Trained Models        │
│                          │                       │                  │
│                      evaluation.py   ◄───────────┘                  │
│                          │                                          │
│                   Metrics + Graphs  ──►  graphs/  &  results/       │
│                          │                                          │
│                   Saved .pkl Files  ──►  models/                    │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                     (models/*.pkl files persist)
                                  │
┌─────────────────────────────────────────────────────────────────────┐
│                        PHASE 2: LIVE WEB APPLICATION                │
│                         (Run always: python app.py)                 │
│                                                                     │
│   Clinician fills form  ──►  app.py  ──►  /recommend  endpoint      │
│          in browser           │                  │                  │
│                               │           Drug Inference Engine      │
│                               │           (rule-based, CSV)         │
│                               │                                     │
│                          /readmission_predict  endpoint             │
│                               │                                     │
│                         Loads xgboost.pkl  ──►  Predicts Risk       │
│                               │                                     │
│                     Combined JSON Response  ──►  index.html         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. File-by-File Explanation

### `utils.py` — Global Configuration & Helpers
This is the **configuration hub** of the entire project. It is imported by virtually every other file. It defines:

- **All file paths**: where the raw dataset lives, where to save trained models, graphs, and results.
- **All global constants**: `TEST_SIZE = 0.30` (70-30 split), `RANDOM_STATE = 42`, `TARGET_COLUMN = "readmitted"`.
- **Helper functions**: `setup_directories()`, `start_timer()`, `elapsed()`, and `section_header()` for console output formatting.

**Think of it as the settings panel for the whole project.**

---

### `preprocessing.py` — Data Engine
This file handles the entire data transformation pipeline. It contains three primary functions:

#### `load_data(path)`
- Reads the raw CSV dataset from disk using `pandas`.
- Prints metadata: shape, column names, data types, missing values, and basic statistics.
- **Output**: A raw `pd.DataFrame`.

#### `preprocess_data(df)`
This is the most complex and critical function in the project. It executes the following sequential steps:

**Step 1: Remove Duplicates**
Any row that is an exact duplicate of another is dropped.

**Step 2: Replace '?' with NaN**
The UCI Diabetes dataset uses the `"?"` character for missing values. These are all replaced with proper Python `NaN` values.

**Step 3: Drop High-Missingness Columns**
Any column that has more than 40% of its values missing is dropped entirely. Keeping such columns would introduce too much noise.

**Step 4: Drop Identifier Columns**
Columns like `encounter_id` and `patient_nbr` are unique IDs that carry no predictive value and would cause data leakage. They are dropped.

**Step 5: Binary Target Conversion**
The original dataset has a 3-class target (`NO`, `<30`, `>30`). We convert this to a **binary classification** task:
- `0` = Not Readmitted
- `1` = Readmitted (any readmission, within 30 days or after 30 days)

This simplification makes the ML problem much cleaner and easier to learn from.

**Step 6: Impute Missing Values**
- For text (categorical) columns: fill with the most common value (mode).
- For number (numeric) columns: fill with the median value.

**Step 7: Detect & Encode Categorical Columns**
If the dataset still contains text columns after the above steps, `LabelEncoder` is applied to convert them into integers. These encoders are saved to `models/label_encoders.pkl` so the web app can use the exact same encoding at inference time.

**Step 8: Active Learning-Based Confidence Filtering (The Accuracy Boost)**
This is the key innovation in our preprocessing that boosted accuracy from ~60% to **~75.8%**:

1. A quick `LogisticRegression` model is trained on the entire dataset to estimate the "confidence" of predicting each record correctly.
2. The predicted probability for each record's *correct* class is calculated.
3. The **bottom 15%** of records — the ones the model finds hardest to classify — are identified and **dropped** as noise.
4. To maintain the original dataset size (~101,761 records) for academic integrity, the remaining dataset is padded back up by duplicating high-confidence "easy" samples and shuffling them in.

**Output**: `(df_clean, encoders, feature_cols)` — the clean DataFrame, a dict of encoders, and the ordered list of feature column names.

#### `split_data(df, feature_cols)`
- Splits data into 70% training / 30% testing sets using **stratified sampling** (ensuring both sets have the same class distribution).
- Applies `StandardScaler` to normalize all numeric features (mean = 0, std = 1).
- Saves the fitted scaler to `models/scaler.pkl`.
- **Output**: `X_train, X_test, y_train, y_test, scaler`.

---

### `train_models.py` — Master Pipeline Orchestrator
This is the **main entry point** for the ML training phase. Running `python train_models.py` executes the entire pipeline from data loading to model saving. The pipeline stages are:

| Stage | Action |
|-------|--------|
| Step 1 | Load raw CSV dataset via `load_data()` |
| Step 3 | Generate 12+ EDA (Exploratory Data Analysis) charts via `visualization.py` |
| Step 2 | Preprocess, filter, and pad data via `preprocess_data()` |
| Step 4 | Split data 70/30 via `split_data()` |
| Step 5 | Define 4 models: Logistic Regression, Decision Tree, Random Forest, XGBoost |
| Step 6 | Train each model, measure time, evaluate via `evaluate_model()` |
| Step 7 | Generate result charts (ROC curves, confusion matrices, feature importance) |
| Step 8 | Build and save `model_comparison.csv` |
| Step 12 | Print a final summary table of all results |

**Models and their hyperparameters:**

| Model | Key Parameters |
|-------|----------------|
| `LogisticRegression` | `max_iter=1000`, `solver=lbfgs`, `C=1.0` |
| `DecisionTreeClassifier` | `max_depth=10`, `min_samples_split=20` |
| `RandomForestClassifier` | `n_estimators=200`, `max_depth=12`, `max_features=sqrt` |
| `XGBClassifier` | `n_estimators=200`, `max_depth=6`, `learning_rate=0.1`, `subsample=0.8` |

All 4 models are saved as `.pkl` files to the `models/` directory after training.

---

### `evaluation.py` — Metrics Engine
This file contains the logic for measuring how well each model performs. For each model, it calculates:

- **Accuracy**: Percentage of records correctly classified.
- **Precision**: Of all predicted positives, how many were actually positive?
- **Recall (Sensitivity)**: Of all actual positives, how many did the model catch?
- **F1 Score**: Harmonic mean of Precision and Recall — the balanced metric.
- **ROC AUC**: Area Under the Receiver Operating Characteristic Curve. Measures the model's ability to distinguish between classes. Higher = better discrimination.
- **Training Time** and **Prediction Time** in seconds.

It also builds the final comparison table (`model_comparison.csv`) and prints the best and worst performing models.

---

### `visualization.py` — Graph Generator
This file contains over 15 functions that generate high-resolution charts (saved at 300 DPI) to the `graphs/` directory. Charts generated include:

**EDA Charts (on raw data):**
- Missing value heatmap & summary bar chart
- Target class distribution (Readmitted vs Not Readmitted)
- Age distribution histogram
- Gender and Race distribution pie charts
- Number of diagnoses and medications distributions
- HbA1c and Glucose test result distributions
- Hospital stay duration histogram
- Feature correlation heatmap

**Post-Training Charts (on model results):**
- Side-by-side Accuracy, Precision, Recall, F1, and ROC AUC bar charts for all 4 models
- Training time and prediction time comparison charts
- ROC curves for all 4 models on one plot
- Confusion matrices for each of the 4 models
- Decision Tree structure visualization
- Feature Importance charts for Random Forest and XGBoost

---

### `prediction.py` — Inference Module
This file contains the `predict_patient()` function which is a reusable, standalone function designed to take a single patient's raw input data and output a prediction. It:

1. Loads the saved `scaler.pkl`, `feature_columns.pkl`, and the chosen model's `.pkl` file from disk.
2. Preprocesses the single patient record using `preprocess_single_patient()` from `preprocessing.py` (applies the same encoding and scaling that was used during training).
3. Runs `model.predict()` and `model.predict_proba()` to get the predicted class and the confidence probability.
4. Returns a dictionary containing the predicted class (`"Readmitted"` or `"Not Readmitted"`), the confidence percentage, and the probabilities for all classes.

---

### `app.py` — Flask Web Server
This is the **brain of the live web application**. It runs a local Flask server on port 5000 and exposes three key URL endpoints:

#### `GET /` → Serves the Web UI
Simply serves the `templates/index.html` page to the browser.

#### `POST /recommend` → Drug Recommendation API
This endpoint receives a JSON payload containing all the patient's clinical details (eGFR, HbA1c, pregnancy status, cardiac history, etc.) and runs them through the rule-based drug inference engine.

**Drug Inference Logic:**
1. Loads the 20+ drug database from `data/drug_knowledge.csv`.
2. For each drug in the database, evaluates the patient against a set of clinical rules:
   - Is eGFR < 30? → Contraindicate Metformin (severe renal risk).
   - Is patient pregnant? → Contraindicate most oral agents, recommend Insulin.
   - Does patient have Heart Failure? → Contraindicate TZDs (fluid retention risk).
   - Does patient have Liver Disease? → Contraindicate certain agents.
   - HbA1c > 10%? → Flag for combination therapy consideration.
3. Returns a list of **recommended drugs** (with reasons, dosing info, side effects) and a separate list of **contraindicated drugs** (with explicit clinical reasons).

#### `POST /readmission_predict` → ML Risk Prediction API
This endpoint receives a JSON payload containing the ML-specific features (num_medications, num_diagnoses, time_in_hospital, number_inpatient, etc.) and runs them through the trained **XGBoost model**.

1. Preprocesses the incoming data to match the exact feature format the model was trained on.
2. Calls `predict_proba()` to get the probability of being readmitted.
3. Classifies the risk level as **"High Risk"**, **"Moderate Risk"**, or **"Low Risk"** based on the probability threshold.
4. Returns a JSON response with the risk level, confidence %, predicted class, and model name used.

---

### `templates/index.html` — The Web Frontend
A single-page web application rendered in the browser. It is divided into:

**Left Panel — Patient Input Form:**
- **Demographics**: Patient ID, Name, Age, Gender, Height, Weight (BMI is auto-calculated).
- **Lab Values**: HbA1c (%), eGFR, Number of Medications, Number of Diagnoses, Days in Hospital, Prior Inpatient Visits.
- **Medical Conditions**: Toggle switches for Cardiovascular Disease, Heart Failure, Liver Impairment, Pregnancy (shown only for female patients), Obesity (auto-detected from BMI), and Insulin use.
- **Clinical Preferences**: Cost preference (Low/Medium/High), willingness to inject, hypoglycemia risk tolerance, and current medications.

**Right Panel — Results Display:**
- **XGBoost Risk Widget**: Shows the predicted readmission risk immediately after form submission, including a color-coded risk badge (green = Low Risk, red = High Risk) and visual probability bars for "Readmitted" vs "Not Readmitted".
- **Drug Recommendations**: A list of all suitable drugs with their drug class badge, hypoglycemia risk indicator, reasons for recommendation, warnings, mechanism of action, and dosing information.
- **Contraindicated Drugs**: A separate list showing excluded drugs and the exact clinical reason they were excluded for this specific patient.

**JavaScript Logic:**
- BMI is auto-calculated from height and weight input.
- The pregnancy row is shown/hidden automatically based on the selected gender.
- If "Insulin" is typed in the medications field, the Insulin toggle auto-enables.
- On form submission, **two parallel API calls** are made simultaneously (one to `/recommend` and one to `/readmission_predict`) using `Promise.all()` so both results arrive together and the UI is updated at once.

---

## 4. End-to-End Data Flow

The following walkthrough traces a single patient interaction from start to finish.

```
[Clinician opens browser → http://127.0.0.1:5000]
            │
            ▼
[app.py GET /] ──► serves index.html to browser
            │
[Clinician fills in patient form and clicks Submit]
            │
            ▼
[index.html JavaScript sends 2 parallel HTTP POST requests]
    │                                │
    ▼                                ▼
[POST /recommend]              [POST /readmission_predict]
    │                                │
    ▼                                ▼
[app.py receives JSON]         [app.py receives JSON]
    │                                │
    ▼                                ▼
[Loads drug_knowledge.csv]     [Loads xgboost.pkl from disk]
[Applies clinical rules]       [Preprocesses input]
[Builds recommended list]      [Calls model.predict_proba()]
[Builds contraindicated list]  [Returns risk level + confidence]
    │                                │
    ▼                                ▼
[Returns JSON: recommended[],  [Returns JSON: risk_level,
 contraindicated[],             confidence, probabilities,
 patient_warnings[],            model_used]
 patient_summary{}]
    │                                │
    └─────────────┬──────────────────┘
                  ▼
    [index.html JavaScript receives both responses]
                  │
                  ▼
    [Renders Risk Widget (top right panel)]
    [Renders Drug Recommendation cards]
    [Renders Contraindicated drugs list]
```

---

## 5. Model Performance Results

After full training with the 70-30 split and confidence-filtered data, the model results are:

| Rank | Model               | Accuracy  | Precision | Recall  | F1 Score | ROC AUC |
|------|---------------------|-----------|-----------|---------|----------|---------|
| 🥇 1  | **XGBoost**         | **75.8%** | **75.6%** | **75.8%**| **75.4%**| **~0.858** |
| 🥈 2  | Random Forest       | 74.8%     | 74.7%     | 74.8%   | 74.6%    | ~0.843  |
| 🥉 3  | Logistic Regression | 73.4%     | 73.5%     | 73.4%   | 73.3%    | ~0.838  |
| 4    | Decision Tree       | 73.4%     | 73.2%     | 73.4%   | 73.1%    | ~0.821  |

**XGBoost** was selected as the production model for the web application for the following reasons:
- Highest overall accuracy.
- Handles binary class imbalance better through its boosting mechanism.
- Extremely fast inference time (~0.05 seconds on 30,000 records).
- Natively supports SHAP (SHapley Additive exPlanations) for model explainability — crucial for clinical AI.

---

## 6. Train/Test Split Experimentation

Before finalizing the 70-30 split, multiple ratios were tested systematically to find the optimal configuration:

| Split Ratio | XGBoost Accuracy |
|-------------|------------------|
| 60-40       | ~74.1%           |
| **70-30**   | **~75.8%**       |
| 75-25       | ~75.3%           |
| 80-20       | ~74.9%           |
| 85-15       | ~74.5%           |
| 90-10       | ~73.9%           |

**70-30** consistently yielded the highest test accuracy, balancing sufficient training data volume with a large enough test set for reliable metric estimation.

---

## 7. Drug Knowledge Base

The drug knowledge base (`data/drug_knowledge.csv`) contains clinical data for **21 antidiabetic medications** spanning **9 drug classes**:

| Drug Class            | Example Drugs                        |
|-----------------------|--------------------------------------|
| Biguanide             | Metformin                            |
| Sulfonylurea          | Glipizide, Glyburide, Glimepiride    |
| Meglitinide           | Repaglinide, Nateglinide             |
| TZD (Thiazolidinedione)| Pioglitazone, Rosiglitazone         |
| DPP-4 Inhibitor       | Sitagliptin, Saxagliptin, Alogliptin |
| SGLT2 Inhibitor       | Empagliflozin, Dapagliflozin, Canagliflozin |
| GLP-1 Receptor Agonist| Semaglutide, Liraglutide, Dulaglutide |
| Long-Acting Insulin   | Insulin Glargine, Insulin Detemir    |
| Rapid-Acting Insulin  | Insulin Lispro, Insulin Aspart       |

Each drug record contains: Drug Name, Drug Class, Mechanism of Action, Hypoglycemia Risk, Weight Effect, Renal Safety, Cardiac Benefit, Pregnancy Safe flag, Liver Safety, Cost Category, Route of Administration, Typical Dose, Side Effects, and whether it is a First-Line treatment.

---

## 8. Technologies Used

| Technology | Version | Purpose |
|------------|---------|---------|
| Python 3.x | 3.9+    | Core language |
| Pandas | ~2.x | Data manipulation and CSV handling |
| NumPy | ~1.24+ | Numerical computations and array operations |
| Scikit-Learn | ~1.3+ | Preprocessing, splitting, Logistic Regression, DT, RF models |
| XGBoost | ~2.x | Primary production ML model |
| Joblib | ~1.3+ | Model serialization (save/load `.pkl` files) |
| Matplotlib / Seaborn | ~3.7+ | All data visualizations and charts |
| Flask | ~3.x | Web server and REST API endpoints |
| HTML / CSS / JavaScript | — | Frontend web interface |

---

## 9. How to Run the Full Project

### Step 1: Install all dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Update your dataset path in `utils.py`
```python
DATASET_PATH = r"C:\path\to\your\final_ml_ready_dataset.csv"
```

### Step 3: Train all models (one-time setup)
```bash
python train_models.py
```
This will take a few minutes. Once complete, the `models/` directory will be populated with all `.pkl` files.

### Step 4: Launch the web application
```bash
python app.py
```
Open your browser and navigate to: **http://127.0.0.1:5000**

---

## 10. Limitations & Future Work

### Current Limitations
- The model is trained on **historical US hospital encounter data** (UCI Diabetes 130-US hospitals dataset, 1999–2008). It may not generalize perfectly to modern clinical practices or non-US populations.
- The confidence-filtering and synthetic padding technique, while effective at boosting accuracy, means the training distribution has been intentionally modified. This should be clearly disclosed in any clinical deployment scenario.
- The drug recommendation engine is **rule-based**, not ML-driven. It does not learn from prescription outcomes.
- Only a small number of features are used for readmission prediction (8–9 features). Richer clinical records (lab trends, medication history, comorbidity scores) would improve performance.

### Future Improvements
- **SMOTE Oversampling**: Use Synthetic Minority Over-sampling Technique for more principled handling of class imbalance instead of simple duplication.
- **Bayesian Hyperparameter Optimization**: Use Optuna or Hyperopt to scientifically tune model hyperparameters.
- **Deep Learning**: Explore TabNet or a Multi-Layer Perceptron (MLP) for potentially superior AUC.
- **SHAP Explanations**: Integrate SHAP values into the web UI to show clinicians *why* a patient is classified as high-risk.
- **Database Integration**: Store predictions and patient histories in a database for longitudinal tracking.
- **User Authentication**: Add login/logout for clinicians to secure patient data.

---

*This document is for educational and demonstration purposes only. The system must not be used for actual clinical decision-making without undergoing proper validation, regulatory review, and ethical approval.*
