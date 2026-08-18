"""
app.py
------
Flask web application for the CDSS Diabetes Mellitus Presentation.

Features
--------
* Accept dummy patient clinical details via a web form.
* Recommend diabetes medications from drug_knowledge.csv using rule-based logic.
* Highlight potential side effects and contraindications based on patient history.
* Expose a clean REST endpoint (/recommend) that returns JSON.

Run with:
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

import os
import sys
import json
import pandas as pd
from flask import Flask, request, jsonify, render_template

# Make sure sibling modules resolve correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import DRUG_KNOWLEDGE_PATH, logger

app = Flask(__name__)

# =============================================================================
# LOAD DRUG KNOWLEDGE AT STARTUP
# =============================================================================
def load_drug_knowledge() -> pd.DataFrame:
    """Load and return the drug knowledge base."""
    if not os.path.exists(DRUG_KNOWLEDGE_PATH):
        raise FileNotFoundError(
            f"Drug knowledge file not found: {DRUG_KNOWLEDGE_PATH}\n"
            "Please confirm the path in utils.py → DRUG_KNOWLEDGE_PATH"
        )
    df = pd.read_csv(DRUG_KNOWLEDGE_PATH, on_bad_lines="skip")
    logger.info(f"Drug knowledge loaded: {len(df)} drugs from {DRUG_KNOWLEDGE_PATH}")
    return df


DRUG_DB: pd.DataFrame = load_drug_knowledge()


# =============================================================================
# DRUG RECOMMENDATION ENGINE
# =============================================================================
def recommend_drugs(patient: dict) -> dict:
    """
    Rule-based drug recommendation engine.

    Patient fields used
    -------------------
    egfr            : float — estimated Glomerular Filtration Rate (kidney function)
    hba1c           : float — current HbA1c level
    pregnant        : bool  — is patient pregnant
    cardiac_history : bool  — has cardiovascular disease history
    obesity         : bool  — BMI >= 30
    hypoglycemia_risk_tolerance : str — 'low' | 'medium' | 'high'
    current_medications : list[str]  — list of drug names patient already takes
    renal_impairment : bool  — severe renal impairment (eGFR < 30)

    Returns
    -------
    dict with:
        recommended   : list of {drug info + reasons + warnings}
        contraindicated : list of {drug name + reason}
        warnings      : list of general patient-level warnings
    """
    df = DRUG_DB.copy()

    egfr                = float(patient.get("egfr", 90))
    hba1c               = float(patient.get("hba1c", 7.0))
    pregnant            = bool(patient.get("pregnant", False))
    cardiac_history     = bool(patient.get("cardiac_history", False))
    obesity             = bool(patient.get("obesity", False))
    renal_impairment    = egfr < 30
    moderate_renal      = 30 <= egfr < 60
    hypo_tolerance      = patient.get("hypoglycemia_risk_tolerance", "low").lower()
    current_meds        = [m.strip().lower() for m in patient.get("current_medications", [])]
    
    # New clinical parameters
    heart_failure       = bool(patient.get("heart_failure", False))
    liver_disease       = bool(patient.get("liver_disease", False))
    willing_to_inject   = bool(patient.get("willing_to_inject", True))
    cost_preference     = patient.get("cost_preference", "medium").lower()

    recommended     = []
    contraindicated = []
    patient_warnings = []

    # ---- Global warnings --------------------------------------------------
    if renal_impairment:
        patient_warnings.append(
            "Severe renal impairment (eGFR < 30): many oral agents are contraindicated. "
            "Insulin is typically preferred."
        )
    if moderate_renal:
        patient_warnings.append(
            "Moderate renal impairment (eGFR 30–60): dose adjustment required for several agents."
        )
    if pregnant:
        patient_warnings.append(
            "Pregnancy: most oral antidiabetic agents are not recommended. "
            "Insulin is the first-choice agent."
        )
    if hba1c > 10:
        patient_warnings.append(
            f"Very high HbA1c ({hba1c}%): consider combination therapy or insulin initiation."
        )

    # ---- Evaluate each drug -----------------------------------------------
    for _, drug in df.iterrows():
        drug_name       = str(drug.get("Drug_Name", ""))
        drug_class      = str(drug.get("Drug_Class", ""))
        mechanism       = str(drug.get("Mechanism", ""))
        first_line      = str(drug.get("First_Line", "No")).strip().lower() == "yes"
        hypo_risk       = str(drug.get("Hypoglycemia_Risk", "Low")).strip().lower()
        weight_effect   = str(drug.get("Weight_Effect", "Neutral"))
        cv_benefit      = str(drug.get("Cardiovascular_Benefit", "Neutral")).strip().lower()
        kidney_benefit  = str(drug.get("Kidney_Benefit", "No")).strip().lower()
        pregnancy_note  = str(drug.get("Pregnancy", "Avoid"))
        renal_adj       = str(drug.get("Renal_Adjustment", "Not required"))
        start_dose      = str(drug.get("Typical_Starting_Dose", ""))
        max_dose        = str(drug.get("Maximum_Dose", ""))
        route           = str(drug.get("Route", "Oral"))
        
        hf_flag         = str(drug.get("Heart_Failure", "Neutral")).strip().lower()
        hepatic_flag    = str(drug.get("Hepatic_Impairment", "Safe")).strip().lower()
        cost_tier       = str(drug.get("Cost_Tier", "Medium")).strip().lower()
        inject_req      = str(drug.get("Injection_Required", "No")).strip().lower() == "yes"

        warnings   = []  # drug-specific warnings for this patient
        reasons    = []  # positive reasons to recommend
        skip       = False
        skip_reason = ""

        # --- CONTRAINDICATION CHECKS --------------------------------------

        # Injection preference
        if not willing_to_inject and inject_req:
            skip = True
            skip_reason = "Patient refuses injectables; this medication requires injection."

        # Pregnancy contraindications
        if pregnant:
            preg_lower = pregnancy_note.lower()
            if any(kw in preg_lower for kw in ["avoid", "contraindicated", "not recommended"]):
                if "insulin" not in drug_name.lower():
                    skip = True
                    skip_reason = f"Contraindicated in pregnancy ({pregnancy_note})."

        # Severe renal impairment
        if renal_impairment and drug_class.lower() == "biguanide":
            skip = True
            skip_reason = "Metformin is contraindicated with severe renal impairment (eGFR < 30) due to lactic acidosis risk."

        if renal_impairment and drug_class.lower() == "sglt2 inhibitor":
            skip = True
            skip_reason = "SGLT2 inhibitors are not effective and generally avoided when eGFR < 30."

        # Hypoglycemia risk vs tolerance
        if hypo_tolerance == "low" and hypo_risk == "high":
            skip = True
            skip_reason = "High hypoglycemia risk drug not suitable for patient with low hypoglycemia tolerance."

        # Heart Failure
        if heart_failure:
            if hf_flag == "contraindicated":
                skip = True
                skip_reason = "Strictly contraindicated in heart failure."
            elif hf_flag == "caution":
                warnings.append("Caution advised in heart failure.")
            elif hf_flag == "beneficial":
                reasons.append("Highly recommended for patients with heart failure.")

        # Liver Disease
        if liver_disease:
            if hepatic_flag == "avoid":
                skip = True
                skip_reason = "Avoid or contraindicated in hepatic impairment."
            elif hepatic_flag == "caution":
                warnings.append("Caution or dose adjustment required in hepatic impairment.")

        # Already on the drug
        if drug_name.lower() in current_meds:
            skip = True
            skip_reason = "Patient is already on this medication."

        if skip:
            contraindicated.append({
                "drug_name":   drug_name,
                "drug_class":  drug_class,
                "reason":      skip_reason,
            })
            continue

        # --- POSITIVE REASON SCORING ---------------------------------------

        if first_line:
            reasons.append("First-line agent for Type 2 Diabetes.")

        # Cardiovascular benefit
        if cardiac_history and cv_benefit in ["yes", "proven", "beneficial", "high"]:
            reasons.append("Has proven cardiovascular benefit — beneficial given patient's cardiac history.")
        elif cardiac_history and cv_benefit == "neutral":
            warnings.append("No proven cardiovascular benefit; consider CV-beneficial agent if available.")

        # Kidney benefit
        if moderate_renal or renal_impairment:
            if kidney_benefit in ["yes", "proven", "nephroprotective"]:
                reasons.append("Nephroprotective — beneficial given patient's reduced renal function.")
            if renal_adj.lower() not in ["not required", "no"]:
                warnings.append(f"Renal dose adjustment required: {renal_adj}.")

        # Weight consideration
        if obesity:
            if "loss" in weight_effect.lower() or "neutral" in weight_effect.lower():
                reasons.append(f"Weight-neutral or weight-reducing — suitable for overweight/obese patient (Weight Effect: {weight_effect}).")
            elif "gain" in weight_effect.lower():
                warnings.append(f"May cause weight gain ({weight_effect}) — consider in context of obesity.")

        # Cost Preference
        if cost_preference == "low" and cost_tier == "high":
            warnings.append("High-cost medication; conflicts with patient's low-cost preference.")
        elif cost_preference == "low" and cost_tier == "low":
            reasons.append("Low-cost / Generic available.")

        # HbA1c level context
        if hba1c > 8.5:
            reasons.append(f"Patient HbA1c is {hba1c}% — potent glucose-lowering agents are preferable.")

        # Hypoglycemia risk note
        if hypo_risk in ["medium", "high"]:
            warnings.append(f"Hypoglycemia risk: {hypo_risk.capitalize()} — monitor blood glucose carefully.")
        else:
            reasons.append(f"Low hypoglycemia risk — safe for patient profile.")

        # Moderate renal adjustment
        if moderate_renal and renal_adj.lower() not in ["not required", "no", ""]:
            warnings.append(f"Moderate renal impairment: {renal_adj}.")

        recommended.append({
            "drug_name":        drug_name,
            "drug_class":       drug_class,
            "mechanism":        mechanism,
            "route":            route,
            "starting_dose":    start_dose,
            "max_dose":         max_dose,
            "weight_effect":    weight_effect,
            "cv_benefit":       drug.get("Cardiovascular_Benefit", "Neutral"),
            "kidney_benefit":   drug.get("Kidney_Benefit", "No"),
            "hypoglycemia_risk":drug.get("Hypoglycemia_Risk", "Low"),
            "pregnancy_note":   pregnancy_note,
            "renal_adjustment": renal_adj,
            "cost_tier":        drug.get("Cost_Tier", "Medium"),
            "reasons":          reasons,
            "warnings":         warnings,
            "is_first_line":    first_line,
            "score":            len(reasons),   # higher = more reasons to prefer
            "effectiveness":    min(98, 70 + (len(reasons) * 6) + (10 if first_line else 0)) # Pseudo-confidence %
        })

    # Sort: first-line first, then by number of positive reasons
    recommended.sort(key=lambda x: (not x["is_first_line"], -x["score"]))
    
    # Generate Patient Summary String
    patient_id = patient.get("patient_id", "N/A")
    name = patient.get("patient_name", "Unknown Patient")
    age = patient.get("patient_age", 0)
    gender = patient.get("patient_gender", "Unknown").capitalize()
    bmi = patient.get("patient_bmi", 0)
    bmi_class = "Obese" if bmi >= 30 else ("Overweight" if bmi >= 25 else "Normal")
    
    summary_text = f"{name} (ID: {patient_id}), {age}yo {gender}, BMI {bmi} ({bmi_class}). "
    summary_text += f"Key Labs: HbA1c {hba1c}%, eGFR {egfr} mL/min. "
    
    constraints = []
    if cardiac_history: constraints.append("Cardiac History")
    if heart_failure: constraints.append("Heart Failure")
    if liver_disease: constraints.append("Hepatic Impairment")
    if pregnant: constraints.append("Pregnant")
    if not willing_to_inject: constraints.append("Refuses Injectables")
    
    if constraints:
        summary_text += "Clinical constraints: " + ", ".join(constraints) + "."

    return {
        "recommended":      recommended,
        "contraindicated":  contraindicated,
        "patient_warnings": patient_warnings,
        "patient_summary_text": summary_text,
        "patient_summary": {
            "eGFR":                 egfr,
            "HbA1c":                hba1c,
            "Pregnant":             pregnant,
            "Cardiac History":      cardiac_history,
            "Obesity":              obesity,
            "Renal Impairment":     renal_impairment,
            "Moderate Renal":       moderate_renal,
            "Hypo Risk Tolerance":  hypo_tolerance,
        },
    }


# =============================================================================
# ROUTES
# =============================================================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/recommend", methods=["POST"])
def recommend():
    """
    POST /recommend
    Body: JSON with patient fields.
    Returns: JSON with recommended drugs and warnings.
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No patient data provided."}), 400

        # Parse boolean fields from form
        data["pregnant"]        = str(data.get("pregnant", "false")).lower() == "true"
        data["cardiac_history"] = str(data.get("cardiac_history", "false")).lower() == "true"
        data["obesity"]         = str(data.get("obesity", "false")).lower() == "true"

        # Safety guard — pregnancy is only biologically applicable to female patients
        gender = str(data.get("gender", "female")).lower()
        if gender not in ("female",):
            data["pregnant"] = False

        # current_medications can be comma-separated string
        meds = data.get("current_medications", "")
        if isinstance(meds, str):
            data["current_medications"] = [m.strip() for m in meds.split(",") if m.strip()]

        result = recommend_drugs(data)
        return jsonify(result)

    except Exception as exc:
        logger.exception("Error in /recommend endpoint")
        return jsonify({"error": str(exc)}), 500


@app.route("/drugs", methods=["GET"])
def list_drugs():
    """Return the full drug knowledge base as JSON."""
    return jsonify(DRUG_DB.to_dict(orient="records"))


@app.route("/model_comparison", methods=["GET"])
def model_comparison():
    """
    GET /model_comparison
    Returns the model comparison table from results/model_comparison.csv as JSON.
    Includes a 'winner' flag and rank for each model.
    """
    try:
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "model_comparison.csv")
        if not os.path.exists(csv_path):
            return jsonify({"error": "Model comparison CSV not found. Run train_models.py first."}), 404

        df = pd.read_csv(csv_path)
        # Add rank column (already sorted by Accuracy descending in train_models.py)
        df["Rank"] = range(1, len(df) + 1)
        df["is_best"] = df["Rank"] == 1

        # Percentage difference from best model
        best_acc = df["Accuracy"].iloc[0]
        df["acc_diff_from_best"] = ((best_acc - df["Accuracy"]) * 100).round(2)

        records = df.to_dict(orient="records")
        return jsonify({
            "models": records,
            "best_model": df.iloc[0]["Model"],
            "best_accuracy": round(float(best_acc), 4),
            "best_roc_auc": round(float(df.iloc[0]["ROC AUC"]), 4),
        })
    except Exception as exc:
        logger.exception("Error in /model_comparison endpoint")
        return jsonify({"error": str(exc)}), 500


@app.route("/readmission_predict", methods=["POST"])
def readmission_predict():
    """
    POST /readmission_predict
    Uses the trained XGBoost model to predict hospital readmission risk.

    Body: JSON with numeric patient features matching the training dataset.
    Returns: predicted class, probabilities, confidence, and human-readable label.
    """
    try:
        import joblib
        import numpy as np

        models_dir   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        xgb_path     = os.path.join(models_dir, "xgboost.pkl")
        features_path = os.path.join(models_dir, "feature_columns.pkl")
        scaler_path  = os.path.join(models_dir, "scaler.pkl")

        if not os.path.exists(xgb_path):
            return jsonify({
                "error": "XGBoost model not found. Run train_models.py first.",
                "available": False,
            }), 404

        data = request.get_json(force=True) or {}

        # Load artifacts
        model        = joblib.load(xgb_path)
        feature_cols = joblib.load(features_path)
        scaler       = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

        # Build feature vector (fill missing with 0)
        X = np.array([[float(data.get(col, 0)) for col in feature_cols]])

        if scaler is not None:
            X = scaler.transform(X)

        # Predict
        pred_index = int(model.predict(X)[0])
        proba      = model.predict_proba(X)[0].tolist()

        class_labels = ["NO (Not Readmitted)", "YES (Readmitted)"]
        risk_levels  = ["Low Risk", "High Risk"]
        risk_colors  = ["#16A34A", "#DC2626"]

        return jsonify({
            "predicted_index":       pred_index,
            "predicted_class":       class_labels[pred_index],
            "risk_level":            risk_levels[pred_index],
            "risk_color":            risk_colors[pred_index],
            "confidence":            round(proba[pred_index] * 100, 1),
            "probabilities": {
                "NO":      round(proba[0] * 100, 1),
                "YES":     round(proba[1] * 100, 1),
            },
            "model_used": "XGBoost (Best Performing Model)",
            "available": True,
        })

    except Exception as exc:
        logger.exception("Error in /readmission_predict endpoint")
        return jsonify({"error": str(exc), "available": False}), 500


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  CDSS Diabetes Web App — Starting")
    print("  Open http://127.0.0.1:5000 in your browser")
    print("=" * 60)
    app.run(debug=True, port=5000)
