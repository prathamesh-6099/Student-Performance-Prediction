"""
src/explain.py
--------------
Explainability layer powered by SHAP (SHapley Additive exPlanations) and What-If Counterfactuals.

Core Responsibilities:
1. Explainer Factory: Constructs TreeExplainer for tree-based models (Random Forest, XGBoost)
   or LinearExplainer for linear baselines (Ridge).
2. JSON-Serializable Explanation Payload: Calculates local SHAP attribution values,
   identifies top 3 negative drivers (pulling score down) and top 3 positive drivers (pushing score up).
3. Ethical Actionable Guardrails:
   - ACTIONABLE vs. NON_ACTIONABLE constants.
   - Suggestions are derived EXCLUSIVELY from actionable behaviors (e.g., Attendance, Study Hours).
   - Non-actionable, fixed, or sensitive attributes (Gender, Parental Education, Family Income) are strictly barred from recommendations.
4. Counterfactual What-If Logic: Simulates real-world improvements on actionable features within
   valid ranges and calculates the model re-predicted score gains.
5. Global Visualizations: Generates beeswarm summary and global feature-importance bar charts.
"""

import sys
from pathlib import Path

# Add src directory to path for robust imports
sys.path.append(str(Path(__file__).parent))

from typing import Dict, Any, List, Union, Tuple, Optional
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge

from labels import convert_score_to_band, DEFAULT_FAIL_THRESHOLD, DEFAULT_PASS_THRESHOLD

# Explicit Feature Boundary Constants (Ethical Explainability Rules)
ACTIONABLE_FEATURES: List[str] = [
    "Hours_Studied",
    "Study_Hours_per_Week",       # Dataset alias
    "Attendance",
    "Attendance_Rate",            # Dataset alias
    "Sleep_Hours",
    "Tutoring_Sessions",
    "Physical_Activity",
    "Extracurricular_Activities"
]

NON_ACTIONABLE_FEATURES: List[str] = [
    "Gender",
    "Family_Income",
    "School_Type",
    "Parental_Education_Level",
    "Distance_from_Home",
    "Learning_Disabilities",
    "Peer_Influence",
    "Parental_Involvement",
    "Previous_Scores",
    "Past_Exam_Scores",           # Dataset alias
    "Internet_Access",
    "Internet_Access_at_Home",     # Dataset alias
    "Student_ID",
    "Pass_Fail"
]


def get_shap_explainer(
    model: Any,
    X_background: Optional[pd.DataFrame] = None
) -> shap.Explainer:
    """
    Factory function returning the appropriate SHAP Explainer based on model architecture.

    Args:
        model: Trained scikit-learn or XGBoost model.
        X_background: Background feature sample (used for LinearExplainer or KernelExplainer).

    Returns:
        Configured SHAP explainer instance.
    """
    if isinstance(model, (RandomForestRegressor, XGBRegressor)) or hasattr(model, "estimators_"):
        # Exact and fast calculation for tree ensembles
        return shap.TreeExplainer(model)
    elif isinstance(model, Ridge):
        if X_background is None:
            raise ValueError("LinearExplainer requires X_background data for computing feature baselines.")
        return shap.LinearExplainer(model, X_background)
    else:
        # Fallback to generic Explainer
        return shap.Explainer(model, X_background)


def simulate_what_if_improvements(
    student_dict: Dict[str, Any],
    model: Any,
    preprocessor: Any,
    feature_names: List[str],
    base_prediction: float
) -> List[Dict[str, Any]]:
    """
    Simulates concrete, realistic improvements strictly across ACTIONABLE features
    and quantifies the estimated score change by re-predicting with the ML model.

    Guardrails:
    - Never modifies NON_ACTIONABLE features.
    - Caps all improvements within realistic bounds (e.g. Attendance <= 100%, Study Hours <= 40).
    - Only suggests changes that yield positive score gains.

    Returns:
        List of actionable suggestion dicts sorted by estimated score improvement.
    """
    suggestions = []

    # 1. Study Hours Improvement (+5 hours, capped at 40 hrs/week)
    study_col = next((c for c in ["Study_Hours_per_Week", "Hours_Studied"] if c in student_dict), None)
    if study_col and study_col in ACTIONABLE_FEATURES:
        curr_val = float(student_dict[study_col])
        if curr_val < 35.0:
            target_val = min(40.0, curr_val + 5.0)
            candidate = student_dict.copy()
            candidate[study_col] = target_val
            proc_df = pd.DataFrame(preprocessor.transform(pd.DataFrame([candidate])), columns=feature_names)
            new_pred = float(model.predict(proc_df)[0])
            delta = round(new_pred - base_prediction, 2)
            if delta > 0.05:
                suggestions.append({
                    "feature": study_col,
                    "current_value": curr_val,
                    "improved_value": target_val,
                    "estimated_score_change": delta,
                    "recommendation": f"Increasing weekly study time by +5 hours (from {curr_val:.0f}h to {target_val:.0f}h) is estimated to change score by {delta:+.2f} points."
                })

    # 2. Attendance Improvement (+10%, capped at 100%)
    att_col = next((c for c in ["Attendance_Rate", "Attendance"] if c in student_dict), None)
    if att_col and att_col in ACTIONABLE_FEATURES:
        curr_val = float(student_dict[att_col])
        if curr_val < 95.0:
            target_val = min(100.0, curr_val + 10.0)
            candidate = student_dict.copy()
            candidate[att_col] = target_val
            proc_df = pd.DataFrame(preprocessor.transform(pd.DataFrame([candidate])), columns=feature_names)
            new_pred = float(model.predict(proc_df)[0])
            delta = round(new_pred - base_prediction, 2)
            if delta > 0.05:
                suggestions.append({
                    "feature": att_col,
                    "current_value": round(curr_val, 1),
                    "improved_value": round(target_val, 1),
                    "estimated_score_change": delta,
                    "recommendation": f"Boosting attendance by +10% (from {curr_val:.1f}% to {target_val:.1f}%) is estimated to change score by {delta:+.2f} points."
                })

    # 3. Extracurricular Activities (If 'No', simulate 'Yes')
    extra_col = "Extracurricular_Activities"
    if extra_col in student_dict and extra_col in ACTIONABLE_FEATURES:
        curr_val = str(student_dict[extra_col])
        if curr_val in ["No", "0", 0]:
            candidate = student_dict.copy()
            candidate[extra_col] = "Yes"
            proc_df = pd.DataFrame(preprocessor.transform(pd.DataFrame([candidate])), columns=feature_names)
            new_pred = float(model.predict(proc_df)[0])
            delta = round(new_pred - base_prediction, 2)
            if delta > 0.05:
                suggestions.append({
                    "feature": extra_col,
                    "current_value": "No",
                    "improved_value": "Yes",
                    "estimated_score_change": delta,
                    "recommendation": f"Engaging in structured extracurricular activities is estimated to change score by {delta:+.2f} points."
                })

    # 4. Tutoring Sessions (+2 sessions, if present)
    tutor_col = "Tutoring_Sessions"
    if tutor_col in student_dict and tutor_col in ACTIONABLE_FEATURES:
        curr_val = float(student_dict[tutor_col])
        if curr_val < 6.0:
            target_val = curr_val + 2.0
            candidate = student_dict.copy()
            candidate[tutor_col] = target_val
            proc_df = pd.DataFrame(preprocessor.transform(pd.DataFrame([candidate])), columns=feature_names)
            new_pred = float(model.predict(proc_df)[0])
            delta = round(new_pred - base_prediction, 2)
            if delta > 0.05:
                suggestions.append({
                    "feature": tutor_col,
                    "current_value": curr_val,
                    "improved_value": target_val,
                    "estimated_score_change": delta,
                    "recommendation": f"Attending +2 tutoring sessions is estimated to change score by {delta:+.2f} points."
                })

    # Sort suggestions by estimated score improvement descending
    suggestions.sort(key=lambda s: s["estimated_score_change"], reverse=True)
    return suggestions


def generate_explanation_payload(
    student_dict: Dict[str, Any],
    model: Optional[Any] = None,
    preprocessor: Optional[Any] = None,
    feature_names: Optional[List[str]] = None,
    explainer: Optional[shap.Explainer] = None,
    fail_threshold: float = DEFAULT_FAIL_THRESHOLD,
    pass_threshold: float = DEFAULT_PASS_THRESHOLD
) -> Dict[str, Any]:
    """
    Produces a complete, JSON-serializable XAI payload for a single student.

    Structure:
    {
        "predicted_score": float,
        "band": "Fail|Borderline|Pass",
        "top_negative_factors": [{"feature", "value", "shap_impact"}],
        "top_positive_factors": [{"feature", "value", "shap_impact"}],
        "actionable_suggestions": [...]
    }

    Args:
        student_dict: Raw student attributes (dict).
        model: Trained regression estimator (loaded from disk if None).
        preprocessor: Fitted ColumnTransformer (loaded from disk if None).
        feature_names: List of column names in order (loaded if None).
        explainer: Fitted SHAP Explainer (instantiated if None).

    Returns:
        JSON-serializable dictionary.
    """
    models_dir = Path("models")

    # Lazy-load artifacts if not provided
    if model is None:
        model = joblib.load(models_dir / "best_model.joblib")
    if preprocessor is None:
        preprocessor = joblib.load(models_dir / "preprocessor.joblib")
    if feature_names is None:
        feature_names = joblib.load(models_dir / "feature_names.joblib")
    if explainer is None:
        explainer = get_shap_explainer(model)

    # 1. Preprocess the single student record
    raw_df = pd.DataFrame([student_dict])
    proc_array = preprocessor.transform(raw_df)
    proc_df = pd.DataFrame(proc_array, columns=feature_names)

    # 2. Model Prediction & Performance Band
    raw_pred = float(model.predict(proc_df)[0])
    predicted_score = round(raw_pred, 2)
    band = convert_score_to_band(predicted_score, fail_threshold, pass_threshold)

    # 3. Compute Local SHAP Attributions
    shap_explanation = explainer(proc_df)
    shap_values = shap_explanation.values[0]

    # 4. Partition Positive and Negative Drivers
    factors = []
    for col_name, shap_val in zip(feature_names, shap_values):
        # Look up original raw value if available in input dictionary
        raw_val = student_dict.get(col_name, proc_df[col_name].iloc[0])
        # Format floating-point raw numbers cleanly
        if isinstance(raw_val, float):
            clean_val = round(raw_val, 2)
        else:
            clean_val = raw_val

        factors.append({
            "feature": col_name,
            "value": clean_val,
            "shap_impact": round(float(shap_val), 3)
        })

    # Negative factors (pulling score down, shap_impact < 0) sorted by impact ascending (most negative first)
    negative_factors = [f for f in factors if f["shap_impact"] < 0]
    negative_factors.sort(key=lambda x: x["shap_impact"])
    top_negative = negative_factors[:3]

    # Positive factors (pushing score up, shap_impact > 0) sorted by impact descending
    positive_factors = [f for f in factors if f["shap_impact"] > 0]
    positive_factors.sort(key=lambda x: x["shap_impact"], reverse=True)
    top_positive = positive_factors[:3]

    # 5. Counterfactual What-If Suggestions (ONLY on actionable features)
    actionable_suggestions = simulate_what_if_improvements(
        student_dict=student_dict,
        model=model,
        preprocessor=preprocessor,
        feature_names=feature_names,
        base_prediction=predicted_score
    )

    payload = {
        "predicted_score": predicted_score,
        "band": band,
        "top_negative_factors": top_negative,
        "top_positive_factors": top_positive,
        "actionable_suggestions": actionable_suggestions
    }

    return payload


def save_global_shap_figures(
    explainer: shap.Explainer,
    X_test_proc: pd.DataFrame,
    reports_dir: Union[str, Path] = "reports/figures"
) -> Tuple[Path, Path]:
    """
    Computes SHAP values on the test set and exports high-resolution global summary
    beeswarm and importance bar charts.

    Returns:
        Tuple of (beeswarm_path, bar_path).
    """
    rep_path = Path(reports_dir)
    rep_path.mkdir(parents=True, exist_ok=True)

    print("[Explainability] Computing global SHAP values across test dataset...")
    shap_values = explainer(X_test_proc)

    # 1. Global Beeswarm Summary Plot
    beeswarm_path = rep_path / "05_shap_summary_beeswarm.png"
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values.values, X_test_proc, show=False)
    plt.title("SHAP Global Feature Impact (Beeswarm)", fontsize=13, weight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=300)
    plt.close()
    print(f"[Explainability] Beeswarm summary saved to: {beeswarm_path.resolve()}")

    # 2. Global Mean |SHAP| Importance Bar Chart
    bar_path = rep_path / "06_shap_global_bar.png"
    plt.figure(figsize=(10, 5))
    shap.summary_plot(shap_values.values, X_test_proc, plot_type="bar", show=False)
    plt.title("Global Feature Importance (Mean |SHAP Value|)", fontsize=13, weight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(bar_path, dpi=300)
    plt.close()
    print(f"[Explainability] Feature importance bar plot saved to: {bar_path.resolve()}")

    return beeswarm_path, bar_path


if __name__ == "__main__":
    models_dir = Path("models")
    if not (models_dir / "best_model.joblib").exists():
        print("Model artifacts not found. Please run src/train.py first.")
        sys.exit(1)

    # Load artifacts
    model = joblib.load(models_dir / "best_model.joblib")
    preprocessor = joblib.load(models_dir / "preprocessor.joblib")
    feature_names = joblib.load(models_dir / "feature_names.joblib")
    X_test_proc = joblib.load(models_dir / "X_test_proc.joblib")
    X_test_raw = joblib.load(models_dir / "X_test_raw.joblib")
    y_test = joblib.load(models_dir / "y_test.joblib")

    # Instantiate explainer
    explainer = get_shap_explainer(model, X_test_proc)

    # 1. Generate and save global SHAP figures
    print("=" * 70)
    print("GENERATING GLOBAL SHAP FIGURES")
    print("=" * 70)
    save_global_shap_figures(explainer, X_test_proc)

    # 2. Select 3 diverse sample students (Low / Fail, Borderline / Medium, High / Pass)
    print("\n" + "=" * 70)
    print("TESTING INDIVIDUAL EXPLANATION PAYLOADS ON 3 SAMPLE STUDENTS")
    print("=" * 70)

    # Score sort to select low, medium, and high student indices
    sorted_indices = y_test.sort_values().index
    low_idx = sorted_indices[0]                    # lowest score student
    mid_idx = sorted_indices[len(sorted_indices) // 2]  # median score student
    high_idx = sorted_indices[-1]                  # highest score student

    sample_cases = [
        ("STUDENT A (Low Performance)", X_test_raw.loc[low_idx].to_dict(), y_test.loc[low_idx]),
        ("STUDENT B (Borderline/Average Performance)", X_test_raw.loc[mid_idx].to_dict(), y_test.loc[mid_idx]),
        ("STUDENT C (High Performance)", X_test_raw.loc[high_idx].to_dict(), y_test.loc[high_idx]),
    ]

    for label, student_data, actual_score in sample_cases:
        print(f"\n--- {label} (Actual Score: {actual_score}) ---")
        payload = generate_explanation_payload(
            student_dict=student_data,
            model=model,
            preprocessor=preprocessor,
            feature_names=feature_names,
            explainer=explainer
        )
        print(json.dumps(payload, indent=2))
