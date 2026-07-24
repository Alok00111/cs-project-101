# Clinical Decision Support System (CDSS)
# Type 2 Diabetes Mellitus — ML Pipeline & Drug Recommendation Web App

Final Year Computer Science Project

---

## Project Structure

```
Diabetes_Project/
├── train_models.py        # Master pipeline script (run this first)
├── preprocessing.py       # Data loading, cleaning, encoding, splitting
├── evaluation.py          # Metrics, confusion matrix, comparison table
├── visualization.py       # All EDA + result graphs (matplotlib only)
├── prediction.py          # Reusable predict_patient() function
├── utils.py               # Constants, paths, helpers
├── app.py                 # Flask web app (drug recommendation)
├── requirements.txt
├── README.md
├── templates/
│   └── index.html         # Presentation web frontend
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
DRUG_KNOWLEDGE_PATH = r"C:\path\to\drug_knowledge.csv"
```

---

## Running the ML Pipeline

```bash
cd Diabetes_Project
python train_models.py
```

This will:
- Load and preprocess the 100k-row dataset
- Generate 25+ EDA and result visualizations → `graphs/`
- Train Logistic Regression, Decision Tree, Random Forest, XGBoost
- Evaluate all models with Accuracy, Precision, Recall, F1, ROC AUC
- Save all trained models → `models/`
- Export comparison table → `results/model_comparison.csv`
- Print final summary to console

---

## Running the Web App (Presentation)

```bash
python app.py
```

Open http://127.0.0.1:5000 in your browser.

Enter a dummy patient's details and click **Get Drug Recommendations**.

The system will:
- Recommend diabetes drugs from the drug knowledge base
- Flag contraindications (pregnancy, renal impairment, hypoglycemia risk)
- Show potential side effects and dose information
- List excluded drugs with specific reasons

---

## Models Trained

| Model               | Algorithm         |
|---------------------|-------------------|
| Logistic Regression | sklearn LR        |
| Decision Tree       | sklearn DT        |
| Random Forest       | sklearn RF        |
| XGBoost             | xgboost XGBClassifier |

---

## Target Variable

`readmitted` — 3 classes:
- `NO`  — Not readmitted
- `>30` — Readmitted after 30 days
- `<30` — Readmitted within 30 days

---

## Drug Knowledge Base

20 drugs across 9 classes:
- Biguanide, Sulfonylurea, Meglitinide, TZD
- DPP-4 Inhibitor, SGLT2 Inhibitor, GLP-1 Receptor Agonist
- Long Acting Insulin, Rapid Acting Insulin

---

## Notes

- The web app **does not require the trained ML models** — it works independently using the drug knowledge CSV.
- The `predict_patient()` function in `prediction.py` can be imported directly into a Flask/FastAPI backend after training.
- All graphs are saved at 300 DPI and are suitable for inclusion in a research paper.
