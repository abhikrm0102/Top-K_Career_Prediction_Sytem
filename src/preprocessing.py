
"""
preprocessing.py
-----------------
Running: -
.\.venv\Scripts\python.exe -m src.preprocessing
-----------------
.\.venv\Scripts\python.exe -m pip install pandas numpy scikit-learn openpyxl
-----------------
Data cleaning, feature selection, train/test splitting, and
machine-learning preprocessing for the career prediction project.

Important:
1. Personally identifiable information is excluded from features.
2. RIASEC-derived features are excluded from career_data by default
   because they may have been used to generate the target labels.
3. RIASEC questionnaire items and derived scores are excluded from
   career_interest_trends by default to avoid circular prediction.
4. Imputers, encoders, and scalers must be fitted on training data only.
"""

import re
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

TARGET_COLUMNS = {
    "student_personality":
        "Which of the following career fields are you most interested in?",
    "career_data": "Career",
    "career_interest_trends": "career_interest_code",
}

# These columns must never be used as predictive features.
PERSONAL_COLUMNS = {
    "id",
    "email",
    "name",
    "participant_id",
}

# Potentially circular features in career_data.
RIASEC_COLUMNS = {
    "r_score",
    "i_score",
    "a_score",
    "s_score",
    "e_score",
    "c_score",
}

# Default test-set proportion.
TEST_SIZE = 0.20
RANDOM_STATE = 42


# ---------------------------------------------------------
# 1. General cleaning functions
# ---------------------------------------------------------

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Remove accidental whitespace from column names."""

    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def normalize_text(value):
    """
    Normalize whitespace in text values.

    Example:
        '  Software   Engineering ' -> 'Software Engineering'
    """

    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()

    return value


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize whitespace in all string/object columns."""

    df = df.copy()

    for column in df.select_dtypes(
        include=["object", "string", "category"]
    ).columns:
        df[column] = df[column].map(normalize_text)

        # Treat empty strings as missing values.
        df[column] = df[column].replace("", np.nan)

    return df


def normalize_target_labels(y: pd.Series) -> pd.Series:
    """Clean target labels without changing their meaning."""

    y = y.copy()

    return y.map(
        lambda value: normalize_text(value)
        if isinstance(value, str)
        else value
    )


def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows and report how many were removed."""

    original_count = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = original_count - len(df)

    if removed:
        print(f"Removed {removed} duplicate row(s).")

    return df


# ---------------------------------------------------------
# 2. Feature selection and dataset preparation
# ---------------------------------------------------------

def prepare_dataset(
    df: pd.DataFrame,
    dataset_key: str,
    target_column: Optional[str] = None,
    allow_riasec_features: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Clean a dataset and separate its features (X) and target (y).

    Parameters
    ----------
    df:
        DataFrame returned by data_loader.py.
    dataset_key:
        One of:
        'student_personality',
        'career_data',
        'career_interest_trends'.
    target_column:
        Optional override for the target column.
    allow_riasec_features:
        If True, permits RIASEC features in datasets where they
        might be related to the target. Use only for a clearly
        documented sensitivity experiment.

    Returns
    -------
    X, y:
        Clean feature DataFrame and target Series.
    """

    supported_datasets = set(TARGET_COLUMNS)

    if dataset_key not in supported_datasets:
        raise ValueError(
            f"Unknown dataset_key: {dataset_key!r}. "
            f"Expected one of {sorted(supported_datasets)}."
        )

    df = clean_column_names(df)
    df = clean_text_columns(df)
    df = remove_duplicate_rows(df)

    target = target_column or TARGET_COLUMNS[dataset_key]

    if target not in df.columns:
        raise KeyError(
            f"Target column {target!r} was not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    # A missing target cannot be used for supervised learning.
    df = df.dropna(subset=[target]).copy()

    # Normalize labels, including whitespace variants.
    y = normalize_target_labels(df[target])

    # Keep only rows with a usable target.
    valid_target = y.notna()
    df = df.loc[valid_target].copy()
    y = y.loc[valid_target].reset_index(drop=True)

    # Remove the target from the feature table.
    X = df.drop(columns=[target]).reset_index(drop=True)

    # Exclude direct identifiers and personal information.
    personal_columns = [
        column
        for column in X.columns
        if str(column).strip().lower() in PERSONAL_COLUMNS
    ]

    if personal_columns:
        X = X.drop(columns=personal_columns)
        print(
            f"[{dataset_key}] Excluded personal/identifier columns: "
            f"{personal_columns}"
        )

    # Dataset-specific leakage prevention.
    if (
        dataset_key == "career_data"
        and not allow_riasec_features
    ):
        leakage_columns = [
            column
            for column in X.columns
            if column.strip().lower() in RIASEC_COLUMNS
        ]

        if leakage_columns:
            X = X.drop(columns=leakage_columns)

        print(
            "[career_data] RIASEC score columns are excluded "
            "from the default feature set."
        )

    if (
        dataset_key == "career_interest_trends"
        and not allow_riasec_features
    ):
        # The 39 questionnaire items and their totals/averages may
        # encode the same information used to assign the target.
        # Exclude them by default. Retain only non-questionnaire
        # features such as demographics, if available.
        questionnaire_columns = []

        for column in X.columns:
            name = str(column).strip().lower()

            is_question_item = (
                name.startswith("item_")
                or name.startswith("question_")
                or re.fullmatch(r"q\d+", name) is not None
            )

            is_riasec_derived = (
                any(
                    token in name
                    for token in [
                        "r_score", "i_score", "a_score",
                        "s_score", "e_score", "c_score",
                        "total", "average", "avg",
                    ]
                )
                or name in RIASEC_COLUMNS
            )

            if is_question_item or is_riasec_derived:
                questionnaire_columns.append(column)

        if questionnaire_columns:
            X = X.drop(columns=questionnaire_columns)

        warnings.warn(
            "career_interest_trends contains a target that may be "
            "derived from its RIASEC questionnaire. The remaining "
            "demographic-only features are not necessarily useful "
            "predictors. Do not interpret high performance from "
            "questionnaire-derived features as independent career "
            "prediction.",
            UserWarning,
        )

    # Remove columns that contain no information.
    all_missing = X.columns[X.isna().all()].tolist()

    if all_missing:
        print(
            f"[{dataset_key}] Removed entirely missing columns: "
            f"{all_missing}"
        )
        X = X.drop(columns=all_missing)

    # Avoid duplicate column names causing ambiguous processing.
    if X.columns.duplicated().any():
        duplicate_names = X.columns[
            X.columns.duplicated()
        ].tolist()

        raise ValueError(
            f"Duplicate feature column names found: {duplicate_names}"
        )

    if X.shape[1] == 0:
        raise ValueError(
            f"No usable features remain for dataset {dataset_key!r}."
        )

    if y.nunique() < 2:
        raise ValueError(
            f"Dataset {dataset_key!r} has fewer than two target classes."
        )

    print(f"\nDataset: {dataset_key}")
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Number of target classes: {y.nunique()}")
    print("Target distribution:")
    print(y.value_counts(dropna=False))

    return X, y


# ---------------------------------------------------------
# 3. Build a machine-learning preprocessing pipeline
# ---------------------------------------------------------

def build_preprocessor(
    X: pd.DataFrame,
) -> ColumnTransformer:
    """
    Create preprocessing pipelines for numeric and categorical data.

    The returned transformer is NOT fitted. Fit it on training
    features only, then use it to transform validation/test features.
    """

    numeric_columns = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_columns = [
        column for column in X.columns
        if column not in numeric_columns
    ]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    transformers = []

    if numeric_columns:
        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    if not transformers:
        raise ValueError("No numeric or categorical features were found.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


# ---------------------------------------------------------
# 4. Leakage-safe train/test split
# ---------------------------------------------------------

def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """
    Split the raw features before fitting any imputer, scaler,
    encoder, or feature-selection procedure.

    Stratification preserves class proportions when possible.
    """

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    class_counts = y.value_counts()
    stratify = y if class_counts.min() >= 2 else None

    if stratify is None:
        warnings.warn(
            "At least one target class has fewer than two samples. "
            "The split will not be stratified.",
            UserWarning,
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------
# 5. Convenience function for preparing a complete dataset
# ---------------------------------------------------------

def preprocess_dataset(
    df: pd.DataFrame,
    dataset_key: str,
    target_column: Optional[str] = None,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    allow_riasec_features: bool = False,
) -> dict:
    """
    Prepare X and y, split the data, and build an unfitted
    preprocessing transformer.

    The transformer must be fitted on X_train only.
    """

    X, y = prepare_dataset(
        df=df,
        dataset_key=dataset_key,
        target_column=target_column,
        allow_riasec_features=allow_riasec_features,
    )

    X_train, X_test, y_train, y_test = split_dataset(
        X=X,
        y=y,
        test_size=test_size,
        random_state=random_state,
    )

    preprocessor = build_preprocessor(X_train)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "feature_columns": X.columns.tolist(),
        "target_name": target_column or TARGET_COLUMNS[dataset_key],
        "dataset_key": dataset_key,
    }


# ---------------------------------------------------------
# 6. Example usage
# ---------------------------------------------------------

if __name__ == "__main__":
    try:
        from .data_loader import load_dataset
    except ImportError:
        from data_loader import load_dataset

    dataset_keys = [
        "student_personality",
        "career_data",
        "career_interest_trends",
    ]

    for key in dataset_keys:
        print("\n" + "=" * 65)
        print(f"Preparing {key}")
        print("=" * 65)

        try:
            raw_df = load_dataset(key)

            result = preprocess_dataset(
                df=raw_df,
                dataset_key=key,
                test_size=0.20,
                random_state=42,
            )

            print("Training feature shape:", result["X_train"].shape)
            print("Testing feature shape:", result["X_test"].shape)
            print("Training target shape:", result["y_train"].shape)
            print("Testing target shape:", result["y_test"].shape)

            # IMPORTANT:
            # Fit only on training data.
            transformer = result["preprocessor"]

            X_train_transformed = transformer.fit_transform(
                result["X_train"]
            )

            # Test data uses the training-fitted transformer.
            X_test_transformed = transformer.transform(
                result["X_test"]
            )

            print(
                "Transformed training shape:",
                X_train_transformed.shape,
            )
            print(
                "Transformed testing shape:",
                X_test_transformed.shape,
            )

        except Exception as exc:
            print(f"Could not process {key}: {exc}")
