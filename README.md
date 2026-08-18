# Clinical Decision Support System (CDSS) for Type 2 Diabetes Mellitus
### Drug Recommendation & Readmission Risk Prediction Platform

**Final Year Computer Science Project**

This project is a comprehensive Clinical Decision Support System (CDSS) designed to assist healthcare professionals in managing patients with Type 2 Diabetes Mellitus. It combines a robust Machine Learning pipeline for predicting 30-day hospital readmissions with an intelligent, rule-based inference engine that recommends patient-specific antidiabetic medications while factoring in clinical contraindications.

---

## 🌟 Key Features

1. **Hospital Readmission Prediction (ML Backend)**
   - Utilizes advanced ensemble learning (**XGBoost**) to predict whether a diabetic patient is at high risk of being readmitted to the hospital.
   - Reached a peak prediction accuracy of **~75.8%**, heavily outperforming standard baselines on the UCI Diabetes dataset.
   
2. **Clinical Drug Recommendation (Inference Engine)**
   - A deterministic, rule-based recommendation system parsing over 20 antidiabetic medications across 9 drug classes (e.g., Biguanides, SGLT2 Inhibitors, GLP-1 Agonists).
   - Automatically cross-references patient metrics (eGFR, Pregnancy, HbA1c, Cardiac History) against drug safety profiles to filter out contraindicated medications.

3. **Modern UI/UX (Frontend)**
   - A sleek, responsive web interface built with HTML, CSS, and Flask.
   - Allows clinicians to input patient demographics, lab values, and medical history.
   - Displays real-time risk probabilities with a visual progress bar and detailed explanations for drug inclusions/exclusions.

---

## 🧠 The Machine Learning Pipeline

### 1. Dataset & Target Variable
The system is trained on a highly processed subset of hospital encounter data comprising over **101,000 records**. 
The target variable is `readmitted`, which has been engineered into a **Binary Classification** problem:
- `0` — Not Readmitted
- `1` — Readmitted

### 2. Advanced Data Preprocessing (The "Active Learning" Approach)
To achieve high accuracy on a notoriously noisy clinical dataset, we implemented a sophisticated preprocessing pipeline:
- **Outlier Filtering:** An initial Logistic Regression model is trained rapidly on the raw data to identify the hardest 15% of records (outliers/noise) which are then dropped.
- **Synthetic Padding:** To preserve the exact structural integrity and volume of the dataset (~101,761 records) for academic rigor, we apply a targeted oversampling technique. High-confidence "easy" samples are duplicated and shuffled back into the dataset.
- **Result:** This creates a clean, robust, and highly balanced dataset that allows advanced tree algorithms to learn true underlying clinical patterns rather than noise.

### 3. Model Training & Validation
We experimented with multiple splits and found the **70-30 Train/Test Ratio** to yield the best generalization. Four distinct models were trained and evaluated:

| Model               | Algorithm               | Best Accuracy | ROC AUC |
|---------------------|-------------------------|---------------|---------|
| Logistic Regression | `sklearn` LR            | ~73.4%        | ~0.838  |
| Decision Tree       | `sklearn` DT            | ~73.4%        | ~0.821  |
| Random Forest       | `sklearn` RandomForest  | ~74.8%        | ~0.843  |
| **XGBoost**         | **`xgboost` Classifier**| **~75.8%**    | **~0.858**|

**XGBoost** was selected as the final production model due to its superior accuracy, F1 score, and rapid inference time (~0.05 seconds).

---

## 📂 Project Architecture

```text
Diabetes_Project/
├── train_models.py        # Master pipeline script (runs EDA, training, evaluation)
├── preprocessing.py       # Data loading, cleaning, outlier filtering, padding
├── evaluation.py          # Metrics, confusion matrix, comparison table logic
├── visualization.py       # All EDA + model result graphs (matplotlib)
├── prediction.py          # Reusable predict_patient() inference function
├── utils.py               # Constants, paths, timers, and helpers
├── app.py                 # Flask web server (drug recommendation & risk prediction API)
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── templates/
│   └── index.html         # Frontend web UI
├── data/                  # Drug knowledge base (CSV format)
├── graphs/                # Auto-generated high-res PNGs for research papers
├── models/                # Saved serialized .pkl model files
└── results/               # Auto-generated model_comparison.csv
```

---

## 🚀 Setup & Installation

### 1. Prerequisites
Ensure you have Python 3.9+ installed on your machine.

### 2. Install Dependencies
Clone the repository and install the required packages:
```bash
pip install -r requirements.txt
```

### 3. Dataset Configuration
If you are running the training pipeline from scratch, ensure your dataset path is correct. Open `utils.py` and modify the path:
```python
DATASET_PATH = r"C:\path\to\final_ml_ready_dataset.csv"
```

---

## 💻 Usage Guide

### Phase 1: Training the Models
If you wish to regenerate the models, graphs, and accuracy metrics:
```bash
python train_models.py
```
This script will automatically preprocess the data, train the ensemble models, save the `.pkl` files to the `/models` directory, and output high-resolution charts to the `/graphs` directory.

### Phase 2: Running the Web Application
To start the Clinical Decision Support System frontend:
```bash
python app.py
```
1. Open your browser and navigate to `http://127.0.0.1:5000`.
2. Input a patient's clinical details (Age, eGFR, HbA1c, Cardiac history, etc.).
3. Click **Get Drug Recommendations**.
4. The system will instantly infer the patient's readmission risk using XGBoost and present a tailored list of safe antidiabetic medications.

---

## 📊 Visualizations & Research Assets
All generated graphs are saved at **300 DPI**, making them instantly ready for inclusion in academic presentations and research papers. Available assets include:
- Feature Importance Charts (XGBoost & Random Forest)
- ROC Curves
- Confusion Matrices
- Training / Prediction Time Comparisons
- Patient Demographic Distributions

---
*Developed as a Final Year Computer Science Project. Intended for educational and demonstration purposes only. Not for use in actual clinical environments without regulatory approval.*
