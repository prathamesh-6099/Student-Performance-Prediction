"""
src/preprocess.py
-----------------
Data loading, cleaning, leak-free preprocessing, and feature encoding.

Key Architectural Decisions:
1. Column-preserving encoding (1:1 mapping): Ordinal/Binary encoding is used instead
   of One-Hot encoding so each original feature corresponds to exactly ONE column.
   This guarantees that SHAP attribution values map 1:1 directly to human-understandable
   student factors without fragmented dummy features.
2. Leakage prevention: Any imputation, encoding, and scaling transformers are fitted
   exclusively on the training split after train/test separation.
3. Outlier boundary compliance: Exam scores exceeding 100 are clipped to the valid
   maximum threshold (100) and reported.
"""

from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# Global seed for deterministic execution (Rule 4)
RANDOM_STATE = 42

# Explicit Column Group Definitions (Task 2 Specifications & Dataset Aliases)
ORDINAL_LOW_MED_HIGH = [
    "Parental_Involvement",
    "Access_to_Resources",
    "Motivation_Level",
    "Family_Income",
    "Teacher_Quality"
]

ORDINAL_OTHER_CATEGORIES: Dict[str, List[str]] = {
    # Handles both 3-tier and 4-tier education representations
    "Parental_Education_Level": ["High School", "College", "Bachelors", "Masters", "PhD", "Postgraduate"],
    "Distance_from_Home": ["Near", "Moderate", "Far"],
    "Peer_Influence": ["Negative", "Neutral", "Positive"]
}

BINARY_COLUMNS = [
    "Extracurricular_Activities",
    "Internet_Access",
    "Internet_Access_at_Home",  # Dataset alias
    "Learning_Disabilities",
    "School_Type",
    "Gender"
]

NUMERIC_COLUMNS = [
    "Hours_Studied",
    "Study_Hours_per_Week",     # Dataset alias
    "Attendance",
    "Attendance_Rate",          # Dataset alias
    "Sleep_Hours",
    "Previous_Scores",
    "Past_Exam_Scores",         # Dataset alias
    "Tutoring_Sessions",
    "Physical_Activity"
]

TARGET_COLUMNS = ["Final_Exam_Score", "Exam_Score"]
NON_PREDICTIVE_COLUMNS = ["Student_ID", "Pass_Fail"]


def load_and_clean_data(
    file_path: str | Path,
    clip_upper_limit: float = 100.0
) -> Tuple[pd.DataFrame, pd.Series, str, int]:
    """
    Loads raw CSV data, identifies target, drops non-predictive identifiers,
    and clips any target values exceeding the ceiling of 100.

    Args:
        file_path: Path to the dataset CSV file.
        clip_upper_limit: Upper ceiling for valid exam scores (default: 100.0).

    Returns:
        X: Feature DataFrame.
        y: Target Series.
        target_col: Name of the target column.
        num_clipped: Number of records where target was clipped (> 100).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path.resolve()}")

    df = pd.read_csv(path)

    # Identify target column dynamically
    target_col = None
    for candidate in TARGET_COLUMNS:
        if candidate in df.columns:
            target_col = candidate
            break

    if target_col is None:
        raise ValueError(f"No valid target column found. Expected one of {TARGET_COLUMNS}")

    # Outlier handling: count and clip scores strictly above 100 (e.g. data-entry errors)
    outliers_mask = df[target_col] > clip_upper_limit
    num_clipped = int(outliers_mask.sum())
    if num_clipped > 0:
        # Clip to ensure valid score boundaries while preserving sample size
        df.loc[outliers_mask, target_col] = clip_upper_limit

    # Separate target
    y = df[target_col].copy()

    # Drop non-predictive identifiers and target from feature matrix X
    cols_to_drop = [c for c in NON_PREDICTIVE_COLUMNS + [target_col] if c in df.columns]
    X = df.drop(columns=cols_to_drop).copy()

    return X, y, target_col, num_clipped


def build_preprocessor(feature_columns: List[str]) -> Tuple[ColumnTransformer, List[str]]:
    """
    Constructs an sklearn ColumnTransformer that preserves a 1:1 mapping between
    original features and processed outputs (no one-hot column expansion).

    Pipeline Components:
    - Ordinal & Binary: SimpleImputer(most_frequent) + OrdinalEncoder
    - Numeric: SimpleImputer(median) + StandardScaler
    All fitted strictly on training data to prevent data leakage (Rule 5).

    Args:
        feature_columns: List of feature column names present in the dataset.

    Returns:
        preprocessor: Configured scikit-learn ColumnTransformer.
        ordered_feature_names: Exact order of feature names produced by the transformer.
    """
    transformers = []
    ordered_feature_names: List[str] = []

    # 1. Standard Ordinal features (Low < Medium < High)
    active_low_med_high = [c for c in ORDINAL_LOW_MED_HIGH if c in feature_columns]
    if active_low_med_high:
        low_med_high_order = ["Low", "Medium", "High"]
        ord_lmh_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(
                categories=[low_med_high_order] * len(active_low_med_high),
                handle_unknown="use_encoded_value",
                unknown_value=-1
            ))
        ])
        transformers.append(("ord_low_med_high", ord_lmh_pipe, active_low_med_high))
        ordered_feature_names.extend(active_low_med_high)

    # 2. Specific Ordinal features with custom hierarchy
    for col, categories in ORDINAL_OTHER_CATEGORIES.items():
        if col in feature_columns:
            # Filter category list to those applicable or matching known ordering
            ord_custom_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(
                    categories=[categories],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1
                ))
            ])
            transformers.append((f"ord_{col}", ord_custom_pipe, [col]))
            ordered_feature_names.append(col)

    # 3. Binary categorical features
    active_binary = [c for c in BINARY_COLUMNS if c in feature_columns]
    for b_col in active_binary:
        # Determine explicit binary categories based on column type
        if b_col == "Gender":
            cats = ["Female", "Male"]
        elif b_col == "School_Type":
            cats = ["Public", "Private"]
        else:
            cats = ["No", "Yes"]

        bin_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(
                categories=[cats],
                handle_unknown="use_encoded_value",
                unknown_value=-1
            ))
        ])
        transformers.append((f"bin_{b_col}", bin_pipe, [b_col]))
        ordered_feature_names.append(b_col)

    # 4. Continuous Numeric features
    active_numeric = [c for c in NUMERIC_COLUMNS if c in feature_columns]
    if active_numeric:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())  # Standard scaling ensures linear models (Ridge) converge reliably
        ])
        transformers.append(("numeric", num_pipe, active_numeric))
        ordered_feature_names.extend(active_numeric)

    # Verify 1:1 column preservation
    if len(ordered_feature_names) != len(feature_columns):
        unmapped = set(feature_columns) - set(ordered_feature_names)
        if unmapped:
            raise ValueError(f"Features present in dataset but not mapped in column groups: {unmapped}")

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor, ordered_feature_names


def prepare_train_test_data(
    file_path: str | Path,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer, List[str], pd.DataFrame, pd.DataFrame]:
    """
    Executes leak-free preprocessing:
    1. Loads and cleans raw data.
    2. Splits into train and test sets BEFORE transformer fitting.
    3. Fits ColumnTransformer on train only, transforms both splits.
    4. Returns processed DataFrames with 1:1 feature names alongside raw unscaled test data for SHAP interpretation.

    Returns:
        X_train_proc: Processed training features.
        X_test_proc: Processed test features.
        y_train: Training target.
        y_test: Test target.
        preprocessor: Fitted ColumnTransformer.
        feature_names: List of feature names matching processed columns.
        X_train_raw: Unscaled raw train features (useful for plain-language context).
        X_test_raw: Unscaled raw test features.
    """
    X, y, target_name, num_clipped = load_and_clean_data(file_path)
    print(f"[Preprocessing] Loaded dataset: {X.shape[0]} samples, {X.shape[1]} input features.")
    print(f"[Preprocessing] Target: '{target_name}'. Clipped records (>100): {num_clipped}")

    # Leakage prevention: Split FIRST before any fitting (Rule 5)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Build and fit preprocessor on training data only
    preprocessor, feature_names = build_preprocessor(list(X.columns))
    X_train_array = preprocessor.fit_transform(X_train_raw)
    X_test_array = preprocessor.transform(X_test_raw)

    # Wrap back into DataFrames to retain feature column alignment
    X_train_proc = pd.DataFrame(X_train_array, columns=feature_names, index=X_train_raw.index)
    X_test_proc = pd.DataFrame(X_test_array, columns=feature_names, index=X_test_raw.index)

    print(f"[Preprocessing] Train shape: {X_train_proc.shape}, Test shape: {X_test_proc.shape}")
    print(f"[Preprocessing] Column mapping is 1:1: {X_train_proc.shape[1] == X_train_raw.shape[1]}")

    return X_train_proc, X_test_proc, y_train, y_test, preprocessor, feature_names, X_train_raw, X_test_raw


if __name__ == "__main__":
    # Smoke test module
    data_path = Path("data/student_performance_dataset.csv")
    X_tr, X_te, y_tr, y_te, prep, f_names, _, _ = prepare_train_test_data(data_path)
    print("Features:", f_names)
    print("X_train head:\n", X_tr.head(2))
