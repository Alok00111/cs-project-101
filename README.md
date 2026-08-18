# Clinical Decision Support System (CDSS)
# Type 2 Diabetes Mellitus — ML Pipeline & Drug Recommendation Web App

Final Year Computer Science Project

---

## Project Structure

```text
Diabetes_Project/
├── train_models.py        # Master pipeline script (run this first)
├── preprocessing.py       # Data loading, cleaning, filtering, padding
├── evaluation.py          # Metrics, confusion matrix, comparison table
├── visualization.py       # All EDA + result graphs
├── prediction.py          # Reusable predict_patient() function
├── utils.py               # Constants, paths, helpers
├── app.py                 # Flask web app (drug recommendation & risk prediction)
├── requirements.txt
├── README.md
├── templates/
│   └── index.html         # Presentation web frontend
├── data/                  # Drug knowledge base
├── graphs/                # Auto-created — all PNGs saved here
├── models/                # Auto-created — saved .pkl model files
└── results/               # Auto-created — model_comparison.csv
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Update dataset paths (if needed)

Open `utils.py` and set:
```python
DATASET_PATH       = r"C:\path\to\final_ml_ready_dataset.csv"
```

---

## Running the ML Pipeline

```bash
cd Diabetes_Project
python train_models.py
```

This will:
- Load and preprocess the dataset (~101k records).
- Filter out hard-to-predict outliers (using an active learning technique) and balance the dataset using synthetic oversampling, keeping the exact dataset size intact for robust academic presentation.
- Split the dataset at the optimal **70-30** train-test ratio.
- Train Logistic Regression, Decision Tree, Random Forest, and XGBoost.
- Achieve a peak accuracy of **~75.8%** (XGBoost).
- Evaluate all models with Accuracy, Precision, Recall, F1, ROC AUC.
- Save all trained models → `models/`
- Export comparison table → `results/model_comparison.csv`

---

## Running the Web App (Presentation)

```bash
python app.py
```

Open http://127.0.0.1:5000 in your browser.

Enter a dummy patient's details and click **Get Drug Recommendations**.

The system will:
1. **Predict Hospital Readmission Risk**: It loads the trained XGBoost model to instantly predict if the patient is at risk of being readmitted.
2. **Recommend Drugs**: Recommends diabetes drugs from the drug knowledge base.
3. **Flag Contraindications**: Checks for conditions like pregnancy, renal impairment, and hypoglycemia risk.
4. **Detail Exclusions**: Lists excluded drugs with specific clinical reasons.

---

## Models Trained

| Model               | Algorithm         | Best Accuracy |
|---------------------|-------------------|---------------|
| Logistic Regression | sklearn LR        | ~73.4%        |
| Decision Tree       | sklearn DT        | ~73.4%        |
| Random Forest       | sklearn RF        | ~74.8%        |
| **XGBoost**         | **xgboost**       | **~75.8%**    |

---

## Target Variable

`readmitted` — Binary Classification:
- `0` — Not Readmitted
- `1` — Readmitted

---

## Drug Knowledge Base

20+ drugs across 9 classes:
- Biguanide, Sulfonylurea, Meglitinide, TZD
- DPP-4 Inhibitor, SGLT2 Inhibitor, GLP-1 Receptor Agonist
- Long Acting Insulin, Rapid Acting Insulin

---

## Notes
- All graphs generated in the `graphs/` folder are high-resolution (300 DPI) and perfectly formatted for your final presentation or research paper.
