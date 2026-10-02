"""
src/labels.py
-------------
Categorical classification bands, threshold calibration, and sensitivity analysis.

Project Rule Alignment:
- Converts continuous regression predictions (Exam_Score) into 3 discrete pedagogical tiers:
    1. Fail       : Score < FAIL_THRESHOLD
    2. Borderline : FAIL_THRESHOLD <= Score < PASS_THRESHOLD
    3. Pass       : Score >= PASS_THRESHOLD
- Provides classification reporting, confusion matrix inspection, and threshold sensitivity analysis.
- Reports and diagnoses empty/tiny 'Fail' class scenarios and suggests calibrated thresholds.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# Canonical Default Threshold Constants
DEFAULT_FAIL_THRESHOLD: float = 50.0
DEFAULT_PASS_THRESHOLD: float = 60.0

LABEL_FAIL: str = "Fail"
LABEL_BORDERLINE: str = "Borderline"
LABEL_PASS: str = "Pass"
BAND_LABELS: List[str] = [LABEL_FAIL, LABEL_BORDERLINE, LABEL_PASS]


def convert_score_to_band(
    score: float,
    fail_threshold: float = DEFAULT_FAIL_THRESHOLD,
    pass_threshold: float = DEFAULT_PASS_THRESHOLD
) -> str:
    """
    Maps a single continuous exam score into one of three performance bands.

    Args:
        score: Continuous exam mark.
        fail_threshold: Cutoff below which student is flagged as Fail.
        pass_threshold: Cutoff at or above which student is marked as Pass.

    Returns:
        One of 'Fail', 'Borderline', or 'Pass'.
    """
    if score < fail_threshold:
        return LABEL_FAIL
    elif score < pass_threshold:
        return LABEL_BORDERLINE
    else:
        return LABEL_PASS


def assign_performance_bands(
    scores: Union[pd.Series, np.ndarray, List[float]],
    fail_threshold: float = DEFAULT_FAIL_THRESHOLD,
    pass_threshold: float = DEFAULT_PASS_THRESHOLD
) -> pd.Series:
    """
    Converts an array/Series of continuous exam scores into discrete bands.

    Args:
        scores: Array-like collection of exam scores.
        fail_threshold: Lower threshold (< fail_threshold -> Fail).
        pass_threshold: Upper threshold (>= pass_threshold -> Pass).

    Returns:
        pd.Series of categorical string labels ('Fail', 'Borderline', 'Pass').
    """
    if isinstance(scores, pd.Series):
        s_series = scores
    else:
        s_series = pd.Series(scores)

    return s_series.apply(lambda x: convert_score_to_band(x, fail_threshold, pass_threshold))


def evaluate_classification_bands(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    fail_threshold: float = DEFAULT_FAIL_THRESHOLD,
    pass_threshold: float = DEFAULT_PASS_THRESHOLD
) -> Dict[str, Any]:
    """
    Evaluates classification performance across the 3 performance bands.

    Args:
        y_true: Ground-truth continuous scores.
        y_pred: Model predicted continuous scores.
        fail_threshold: Cutoff for Fail tier.
        pass_threshold: Cutoff for Pass tier.

    Returns:
        Dictionary containing classification report dict, confusion matrix df,
        and count distributions.
    """
    actual_bands = assign_performance_bands(y_true, fail_threshold, pass_threshold)
    pred_bands = assign_performance_bands(y_pred, fail_threshold, pass_threshold)

    # Confusion matrix with fixed labels order: Fail, Borderline, Pass
    cm = confusion_matrix(actual_bands, pred_bands, labels=BAND_LABELS)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual_{lbl}" for lbl in BAND_LABELS],
        columns=[f"Pred_{lbl}" for lbl in BAND_LABELS]
    )

    report_dict = classification_report(
        actual_bands,
        pred_bands,
        labels=BAND_LABELS,
        output_dict=True,
        zero_division=0
    )

    actual_counts = actual_bands.value_counts().reindex(BAND_LABELS, fill_value=0).to_dict()
    pred_counts = pred_bands.value_counts().reindex(BAND_LABELS, fill_value=0).to_dict()

    return {
        "confusion_matrix": cm_df,
        "classification_report": report_dict,
        "actual_counts": actual_counts,
        "pred_counts": pred_counts,
        "fail_recall": report_dict[LABEL_FAIL]["recall"],
        "fail_precision": report_dict[LABEL_FAIL]["precision"],
        "fail_f1": report_dict[LABEL_FAIL]["f1-score"],
        "accuracy": report_dict["accuracy"]
    }


def run_threshold_sensitivity_analysis(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    output_path: Union[str, Path] = "reports/threshold_sensitivity.csv"
) -> pd.DataFrame:
    """
    Runs a multi-threshold sensitivity experiment to quantify how Fail-class
    counts and recall shift under varying cutoff boundaries.

    Configurations Tested:
    1. Default (50 / 60): Canonical standard (<50 Fail, 50-59 Borderline, >=60 Pass).
    2. Sensitivity Low/High (45 / 65): Lenient fail boundary, strict pass boundary.
    3. Sensitivity Mid/High (55 / 65): Standard academic grading curve (45/55/65 scale).
    4. Suggested Calibrated (54 / 60): Tailored for datasets where min score is 50.
    5. Suggested Calibrated (55 / 60): Splits lower 50-59 band evenly into Fail (<55) and Borderline (55-59).

    Returns:
        DataFrame summarizing sensitivity results.
    """
    configs = [
        {"name": "Default (50/60)", "fail_cut": 50.0, "pass_cut": 60.0},
        {"name": "Sensitivity (45/65)", "fail_cut": 45.0, "pass_cut": 65.0},
        {"name": "Sensitivity (55/65)", "fail_cut": 55.0, "pass_cut": 65.0},
        {"name": "Suggested Calibrated (54/60)", "fail_cut": 54.0, "pass_cut": 60.0},
        {"name": "Suggested Calibrated (55/60)", "fail_cut": 55.0, "pass_cut": 60.0},
    ]

    records = []
    for cfg in configs:
        res = evaluate_classification_bands(
            y_true, y_pred, fail_threshold=cfg["fail_cut"], pass_threshold=cfg["pass_cut"]
        )
        records.append({
            "Configuration": cfg["name"],
            "Fail_Threshold": cfg["fail_cut"],
            "Pass_Threshold": cfg["pass_cut"],
            "Actual_Fail_Count": res["actual_counts"][LABEL_FAIL],
            "Actual_Borderline_Count": res["actual_counts"][LABEL_BORDERLINE],
            "Actual_Pass_Count": res["actual_counts"][LABEL_PASS],
            "Pred_Fail_Count": res["pred_counts"][LABEL_FAIL],
            "Fail_Recall": round(res["fail_recall"], 4),
            "Fail_Precision": round(res["fail_precision"], 4),
            "Fail_F1": round(res["fail_f1"], 4),
            "Overall_Accuracy": round(res["accuracy"], 4)
        })

    sensitivity_df = pd.DataFrame(records)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    sensitivity_df.to_csv(out_file, index=False)
    print(f"[Labels] Sensitivity analysis saved to: {out_file.resolve()}")

    return sensitivity_df


if __name__ == "__main__":
    import joblib

    models_dir = Path("models")
    y_test_file = models_dir / "y_test.joblib"
    x_test_file = models_dir / "X_test_proc.joblib"
    best_model_file = models_dir / "best_model.joblib"

    if not (y_test_file.exists() and best_model_file.exists()):
        print("Model artifacts not found. Please run src/train.py first.")
        exit(1)

    y_test = joblib.load(y_test_file)
    X_test_proc = joblib.load(x_test_file)
    model = joblib.load(best_model_file)
    preds = model.predict(X_test_proc)

    print("=" * 70)
    print("DEFAULT THRESHOLDS EVALUATION: Fail (<50), Borderline (50-59), Pass (>=60)")
    print("=" * 70)
    default_res = evaluate_classification_bands(y_test, preds, 50.0, 60.0)

    print("\nActual Test Set Band Counts:")
    for lbl, cnt in default_res["actual_counts"].items():
        print(f"  - {lbl:12}: {cnt} ({cnt / len(y_test) * 100:.1f}%)")

    print("\nPredicted Test Set Band Counts:")
    for lbl, cnt in default_res["pred_counts"].items():
        print(f"  - {lbl:12}: {cnt} ({cnt / len(y_test) * 100:.1f}%)")

    print("\nConfusion Matrix:")
    print(default_res["confusion_matrix"])

    print("\nFail-Class Metrics:")
    print(f"  - Fail Recall    : {default_res['fail_recall']:.4f}")
    print(f"  - Fail Precision : {default_res['fail_precision']:.4f}")
    print(f"  - Fail F1-Score  : {default_res['fail_f1']:.4f}")
    print(f"  - Overall Accuracy: {default_res['accuracy']:.4f}")

    if default_res["actual_counts"][LABEL_FAIL] == 0:
        print("\n[DIAGNOSTIC NOTICE]: The 'Fail' class (<50) contains 0 students!")
        print("Because the dataset's exam scores strictly bottom out at 50, a strict '<50' cutoff")
        print("leaves the Fail bucket completely empty. Borderline absorbs all failing students.")
        print("RECOMMENDATION: Use calibrated thresholds (e.g. Fail < 54 or < 55, Borderline 54/55-59, Pass >= 60).")

    print("\n" + "=" * 70)
    print("THRESHOLD SENSITIVITY CHECK (reports/threshold_sensitivity.csv)")
    print("=" * 70)
    sens_df = run_threshold_sensitivity_analysis(y_test, preds)
    print(sens_df.to_string(index=False))
