"""
src/train.py
------------
Model training, 5-fold CV hyperparameter tuning, model comparison, and artifact saving.

Models Trained:
1. Ridge Regression (linear L2 baseline)
2. Random Forest Regressor (non-linear bagging ensemble)
3. XGBoost Regressor (gradient boosting ensemble)

Outputs:
- reports/model_comparison.csv: Test set performance metrics (RMSE, MAE, R2).
- models/best_model.joblib: Best model selected by test RMSE.
- models/preprocessor.joblib: Fitted ColumnTransformer for inferencing.
- models/feature_names.joblib: List of feature names matching processed inputs.
- models/X_test.joblib, models/y_test.joblib, models/X_test_raw.joblib: Evaluation split for explainability.
"""

import sys
from pathlib import Path

# Add src directory to path to ensure robust local importing
sys.path.append(str(Path(__file__).parent))

from typing import Dict, Any, Tuple
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

from preprocess import prepare_train_test_data, RANDOM_STATE
from labels import (
    evaluate_classification_bands,
    run_threshold_sensitivity_analysis,
    DEFAULT_FAIL_THRESHOLD,
    DEFAULT_PASS_THRESHOLD
)


def run_training_pipeline(
    data_path: str | Path = "data/student_performance_dataset.csv",
    reports_dir: str | Path = "reports",
    models_dir: str | Path = "models"
) -> Tuple[pd.DataFrame, str, Any]:
    """
    Executes end-to-end model training, hyperparameter tuning with 5-fold CV,
    evaluates candidates on test data, and saves artifacts.

    Args:
        data_path: Path to dataset CSV.
        reports_dir: Directory where comparison table is saved.
        models_dir: Directory where model artifacts are persisted.

    Returns:
        comparison_df: Table of performance metrics across all models.
        best_model_name: Name of the top-performing model.
        best_estimator: Fitted estimator instance of the best model.
    """
    reports_path = Path(reports_dir)
    models_path = Path(models_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    models_path.mkdir(parents=True, exist_ok=True)

    # 1. Leak-free data preparation with 80/20 train/test split (Rule 5)
    print("=" * 70)
    print("STEP 1: PREPROCESSING & DATA SPLIT")
    print("=" * 70)
    (
        X_train_proc,
        X_test_proc,
        y_train,
        y_test,
        preprocessor,
        feature_names,
        X_train_raw,
        X_test_raw
    ) = prepare_train_test_data(data_path, test_size=0.2, random_state=RANDOM_STATE)

    # 2. Define Candidate Models and Lightweight Hyperparameter Grids
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    model_configs: Dict[str, Dict[str, Any]] = {
        "Ridge Regression": {
            "estimator": Ridge(random_state=RANDOM_STATE),
            "param_grid": {
                "alpha": [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]
            }
        },
        "Random Forest": {
            "estimator": RandomForestRegressor(random_state=RANDOM_STATE),
            "param_grid": {
                "n_estimators": [50, 100],
                "max_depth": [3, 5, 8, None],
                "min_samples_split": [2, 5]
            }
        },
        "XGBoost": {
            "estimator": XGBRegressor(random_state=RANDOM_STATE, eval_metric="rmse"),
            "param_grid": {
                "n_estimators": [50, 100],
                "max_depth": [2, 3, 4],
                "learning_rate": [0.05, 0.1]
            }
        }
    }

    # 3. Train each model with 5-fold Cross-Validation
    print("\n" + "=" * 70)
    print("STEP 2: MODEL TUNING & TRAINING (5-FOLD CV)")
    print("=" * 70)

    fitted_models = {}
    performance_records = []

    for name, config in model_configs.items():
        print(f"\n--> Tuning {name}...")
        grid_search = GridSearchCV(
            estimator=config["estimator"],
            param_grid=config["param_grid"],
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1
        )
        grid_search.fit(X_train_proc, y_train)

        best_estimator = grid_search.best_estimator_
        fitted_models[name] = best_estimator
        print(f"    Best Params: {grid_search.best_params_}")
        print(f"    Best 5-Fold CV RMSE: {-grid_search.best_score_:.4f}")

        # Evaluate on the held-out test set
        test_preds = best_estimator.predict(X_test_proc)
        rmse = root_mean_squared_error(y_test, test_preds)
        mae = mean_absolute_error(y_test, test_preds)
        r2 = r2_score(y_test, test_preds)

        performance_records.append({
            "Model": name,
            "Test_RMSE": round(rmse, 4),
            "Test_MAE": round(mae, 4),
            "Test_R2": round(r2, 4),
            "Best_Params": str(grid_search.best_params_)
        })

    # 4. Compile Comparison Table
    comparison_df = pd.DataFrame(performance_records).sort_values(by="Test_RMSE", ascending=True).reset_index(drop=True)
    comparison_file = reports_path / "model_comparison.csv"
    comparison_df.to_csv(comparison_file, index=False)
    print("\n" + "=" * 70)
    print("STEP 3: TEST SET PERFORMANCE COMPARISON")
    print("=" * 70)
    print(comparison_df.to_string(index=False))
    print(f"\nComparison table saved to: {comparison_file}")

    # 5. Identify Winner and Save Artifacts
    best_row = comparison_df.iloc[0]
    best_model_name = best_row["Model"]
    best_estimator = fitted_models[best_model_name]

    print("\n" + "=" * 70)
    print(f"STEP 4: SAVING BEST MODEL & PREPROCESSING ARTIFACTS")
    print("=" * 70)

    best_model_path = models_path / "best_model.joblib"
    preprocessor_path = models_path / "preprocessor.joblib"
    feature_names_path = models_path / "feature_names.joblib"

    joblib.dump(best_estimator, best_model_path)
    joblib.dump(preprocessor, preprocessor_path)
    joblib.dump(feature_names, feature_names_path)

    # Also persist test sets for downstream SHAP evaluation & Streamlit demo
    joblib.dump(X_test_proc, models_path / "X_test_proc.joblib")
    joblib.dump(y_test, models_path / "y_test.joblib")
    joblib.dump(X_test_raw, models_path / "X_test_raw.joblib")

    print(f"Saved best model ({best_model_name}) to: {best_model_path}")
    print(f"Saved preprocessor to: {preprocessor_path}")
    print(f"Saved feature names list to: {feature_names_path}")
    print(f"Saved test evaluation splits to: {models_path}")

    # 6. Honest Performance Analysis
    print("\n" + "=" * 70)
    print("STEP 5: HONEST ARCHITECTURAL ANALYSIS")
    print("=" * 70)

    ridge_metrics = comparison_df[comparison_df["Model"] == "Ridge Regression"].iloc[0]
    winner_metrics = best_row

    rmse_diff = ridge_metrics["Test_RMSE"] - winner_metrics["Test_RMSE"]
    r2_diff = winner_metrics["Test_R2"] - ridge_metrics["Test_R2"]

    print(f"WINNER: '{best_model_name}'")
    print(f"  - Test RMSE: {winner_metrics['Test_RMSE']} (vs Ridge: {ridge_metrics['Test_RMSE']}, diff: -{rmse_diff:.4f})")
    print(f"  - Test R2:   {winner_metrics['Test_R2']} (vs Ridge: {ridge_metrics['Test_R2']}, diff: +{r2_diff:.4f})")
    print(f"  - Test MAE:  {winner_metrics['Test_MAE']} (vs Ridge: {ridge_metrics['Test_MAE']})")

    print("\nHonest Assessment:")
    if rmse_diff < 0.2:
        print("  The simple linear Ridge Regression model performed almost identically to the complex")
        print("  tree ensembles. In this regime, Ridge would often be preferred for parsimony and simplicity.")
    else:
        print(f"  {best_model_name} outperforms Ridge Regression by a noticeable margin of {rmse_diff:.2f} points in RMSE")
        print(f"  (an R2 boost from {ridge_metrics['Test_R2']} to {winner_metrics['Test_R2']}).")
        print("  This indicates that non-linear interaction terms (e.g. past scores combined with attendance)")
        print("  are effectively captured by tree-based partitioning. Random Forest / XGBoost is well-justified.")

    # 7. Classification Bands & Threshold Sensitivity (Task 3)
    print("\n" + "=" * 70)
    print("STEP 6: CLASSIFICATION BANDS & THRESHOLD SENSITIVITY")
    print("=" * 70)
    best_preds = best_estimator.predict(X_test_proc)
    band_results = evaluate_classification_bands(
        y_test, best_preds, DEFAULT_FAIL_THRESHOLD, DEFAULT_PASS_THRESHOLD
    )
    print("Default Thresholds Confusion Matrix (Fail < 50, Borderline 50-59, Pass >= 60):")
    print(band_results["confusion_matrix"])
    print(f"Fail Recall: {band_results['fail_recall']:.4f} | Overall Accuracy: {band_results['accuracy']:.4f}")

    sensitivity_file = reports_path / "threshold_sensitivity.csv"
    run_threshold_sensitivity_analysis(y_test, best_preds, output_path=sensitivity_file)

    return comparison_df, best_model_name, best_estimator


if __name__ == "__main__":
    run_training_pipeline()
