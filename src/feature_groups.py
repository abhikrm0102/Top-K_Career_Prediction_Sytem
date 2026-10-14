
"""
feature_groups.py
-----------------
Feature-group definitions and leakage-aware experimental splits
for the Explainable Multi-Source Career Recommendation System.

Design principles:
- Keep datasets separate unless target mappings are justified.
- Exclude identifiers and personally identifiable information.
- Keep potentially label-generating features out of baseline models.
- Split raw rows before fitting preprocessing transformers.
- Support stratified and group-aware holdout splits.
- Make ablation experiments reproducible.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    GroupShuffleSplit,
    train_test_split,
)


# =========================================================
# 1. Global experiment settings
# =========================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

DATASET_KEYS = (
    "student_personality",
    "career_data",
    "career_interest_trends",
)

TARGET_COLUMNS = {
    "student_personality":
        "Which of the following career fields are you most interested in?",
    "career_data": "Career",
    "career_interest_trends": "career_interest_code",
}

IDENTIFIER_NAMES = {
    "id",
    "email",
    "name",
    "participant_id",
    "student_id",
    "record_id",
}

DEMOGRAPHIC_NAMES = {
    "gender",
    "sex",
    "age",
    "age (years)",
    "region",
    "school",
    "semester",
}

RIASEC_NAMES = {
    "r_score",
    "i_score",
    "a_score",
    "s_score",
    "e_score",
    "c_score",
}

RIASEC_CODES = {"r", "i", "a", "s", "e", "c"}


# =========================================================
# 2. Dataset-specific feature groups
# =========================================================

@dataclass(frozen=True)
class Experiment:
    """Definition of one feature-set experiment."""

    name: str
    dataset_key: str
    groups: Tuple[str, ...]
    description: str
    sensitivity_only: bool = False


# Column names are matched case-insensitively after stripping spaces.
# These definitions use the column names established in the project.
KNOWN_FEATURE_GROUPS = {
    "student_personality": {
        "demographics": [
            "Gender",
            "Age (Years)",
            "Which region or state are you from?",
            "(Semester)",
        ],
        # The meanings of columns 1-10 must be verified from the
        # original questionnaire before interpreting them as traits.
        "questionnaire": [str(i) for i in range(1, 11)],
    },

    "career_data": {
        "academic": [
            "Math_Score",
            "Science_Score",
        ],
        "skills": [
            "Programming_Skill",
            "Communication_Skill",
            "Logical_Ability",
        ],
        "riasec": [
            "R_score",
            "I_score",
            "A_score",
            "S_score",
            "E_score",
            "C_score",
        ],
    },

    "career_interest_trends": {
        "demographics": [
            "SEX",
            "SCHOOL",
        ],
        # Questionnaire items and derived totals/averages are
        # detected dynamically because spreadsheet headings vary.
        "questionnaire": [],
        "riasec": [],
    },
}


def _normalise_name(name: object) -> str:
    """Normalize a column name for robust matching."""

    return " ".join(str(name).strip().lower().split())


def _is_identifier(column: object) -> bool:
    """Return True if a column looks like a direct identifier."""

    return _normalise_name(column) in IDENTIFIER_NAMES


def _is_riasec_score(column: object) -> bool:
    """Recognize RIASEC score columns."""

    name = _normalise_name(column)

    if name in RIASEC_NAMES:
        return True

    compact = name.replace(" ", "").replace("-", "_")
    return compact in RIASEC_NAMES



def _is_questionnaire_item(column: object) -> bool:
    """Recognize common questionnaire-item column names."""

    import re

    name = _normalise_name(column)
    compact = name.replace(" ", "").replace("-", "_")

    if name.isdigit():
        return True

    patterns = (
        r"q_?\d+",
        r"item_?\d+",
        r"question_?\d+",
        r"item_[riasec]_\d+",
        r"question_[riasec]_\d+",
    )

    return any(
        re.fullmatch(pattern, compact)
        for pattern in patterns
    )


def _is_derived_riasec_column(column: object) -> bool:
    """
    Identify totals, averages, or derived RIASEC columns.

    This is intentionally conservative and should be checked
    against the actual Excel column headings.
    """

    name = _normalise_name(column)
    compact = name.replace(" ", "_")

    if _is_riasec_score(column):
        return True

    has_riasec_code = any(
        compact.startswith(code + "_")
        or compact.endswith("_" + code)
        for code in RIASEC_CODES
    )

    derived_terms = (
        "total",
        "average",
        "avg",
        "mean",
        "score",
    )

    return has_riasec_code and any(
        term in compact for term in derived_terms
    )


def get_feature_groups(
    X: pd.DataFrame,
    dataset_key: str,
) -> Dict[str, List[str]]:
    """
    Return feature groups restricted to columns actually present in X.

    X must already have its target removed. Identifiers are excluded
    regardless of whether they appear in the predefined groups.
    """

    if dataset_key not in KNOWN_FEATURE_GROUPS:
        raise ValueError(
            f"Unsupported dataset: {dataset_key!r}. "
            f"Expected one of {DATASET_KEYS}."
        )

    actual_columns = list(X.columns)
    normalized_lookup = {
        _normalise_name(column): column
        for column in actual_columns
    }

    groups: Dict[str, List[str]] = {}
    assigned = set()

    def add_group(group_name: str, candidates: Sequence[str]) -> None:
        matched = []

        for candidate in candidates:
            actual = normalized_lookup.get(_normalise_name(candidate))

            if actual is not None and not _is_identifier(actual):
                if actual not in matched:
                    matched.append(actual)

        groups[group_name] = matched
        assigned.update(matched)

    # Match known columns.
    for group_name, candidates in KNOWN_FEATURE_GROUPS[
        dataset_key
    ].items():
        add_group(group_name, candidates)

    # Dynamically identify questionnaire and RIASEC columns.
    if dataset_key == "career_interest_trends":
        groups["questionnaire"] = [
            column for column in actual_columns
            if (
                _is_questionnaire_item(column)
                and not _is_identifier(column)
            )
        ]

        groups["riasec"] = [
            column for column in actual_columns
            if (
                _is_riasec_score(column)
                or _is_derived_riasec_column(column)
            )
            and not _is_identifier(column)
        ]

        assigned.update(groups["questionnaire"])
        assigned.update(groups["riasec"])

    # Include otherwise-unclassified columns for transparency.
    # They are NOT automatically included in baseline experiments.
    groups["other"] = [
        column for column in actual_columns
        if (
            column not in assigned
            and not _is_identifier(column)
        )
    ]

    # Remove any identifier from every group as a final safeguard.
    for group_name in groups:
        groups[group_name] = [
            column for column in groups[group_name]
            if not _is_identifier(column)
        ]

    return groups


def select_features(
    X: pd.DataFrame,
    dataset_key: str,
    group_names: Sequence[str],
) -> pd.DataFrame:
    """Select the union of requested groups, preserving column order."""

    groups = get_feature_groups(X, dataset_key)

    unknown = [
        group for group in group_names
        if group not in groups
    ]

    if unknown:
        raise ValueError(
            f"Unknown feature group(s): {unknown}. "
            f"Available groups: {list(groups)}"
        )

    selected = set()

    for group_name in group_names:
        selected.update(groups[group_name])

    columns = [
        column for column in X.columns
        if column in selected
    ]

    if not columns:
        raise ValueError(
            f"No columns matched groups {list(group_names)} "
            f"for dataset {dataset_key!r}."
        )

    if any(_is_identifier(column) for column in columns):
        raise AssertionError("An identifier entered the feature set.")

    return X.loc[:, columns].copy()


# =========================================================
# 3. Predefined experiments
# =========================================================

EXPERIMENTS = [
    # Student personality dataset
    Experiment(
        name="SP_questionnaire_baseline",
        dataset_key="student_personality",
        groups=("questionnaire",),
        description=(
            "Predict self-reported career preference from "
            "questionnaire responses; validate item meanings first."
        ),
    ),
    Experiment(
        name="SP_demographics_baseline",
        dataset_key="student_personality",
        groups=("demographics",),
        description=(
            "Demographic-only reference model; inspect fairness "
            "and avoid treating demographics as causal explanations."
        ),
    ),
    Experiment(
        name="SP_questionnaire_plus_demographics",
        dataset_key="student_personality",
        groups=("questionnaire", "demographics"),
        description=(
            "Compare questionnaire-only and questionnaire-plus-"
            "demographic performance."
        ),
    ),

    # Career dataset
    Experiment(
        name="CD_academic_baseline",
        dataset_key="career_data",
        groups=("academic",),
        description="Use academic scores only.",
    ),
    Experiment(
        name="CD_skills_baseline",
        dataset_key="career_data",
        groups=("skills",),
        description="Use programming, communication, and logical skills.",
    ),
    Experiment(
        name="CD_academic_plus_skills",
        dataset_key="career_data",
        groups=("academic", "skills"),
        description=(
            "Primary baseline excluding potentially label-generating "
            "RIASEC scores."
        ),
    ),
    Experiment(
        name="CD_riasec_sensitivity",
        dataset_key="career_data",
        groups=("riasec",),
        description=(
            "Sensitivity experiment only; use after documenting "
            "how the Career labels were generated."
        ),
        sensitivity_only=True,
    ),
    Experiment(
        name="CD_all_features_sensitivity",
        dataset_key="career_data",
        groups=("academic", "skills", "riasec"),
        description=(
            "Full-feature sensitivity experiment; not a primary "
            "result until target provenance is verified."
        ),
        sensitivity_only=True,
    ),

    # Career interest trends
    Experiment(
        name="CIT_demographics_exploratory",
        dataset_key="career_interest_trends",
        groups=("demographics",),
        description=(
            "Exploratory demographic-only baseline. Not directly "
            "comparable with other datasets without target mapping."
        ),
    ),
]


def get_experiment(
    experiment_name: str,
    allow_sensitivity: bool = False,
) -> Experiment:
    """Retrieve an experiment by name and enforce sensitivity guard."""

    for experiment in EXPERIMENTS:
        if experiment.name == experiment_name:
            if experiment.sensitivity_only and not allow_sensitivity:
                raise PermissionError(
                    f"{experiment_name!r} is a sensitivity experiment. "
                    "Verify label-generation provenance and call with "
                    "allow_sensitivity=True to run it."
                )

            return experiment

    raise KeyError(f"Experiment not found: {experiment_name!r}")


def list_experiments(
    dataset_key: Optional[str] = None,
) -> List[dict]:
    """Return experiment definitions as serializable dictionaries."""

    return [
        {
            "name": experiment.name,
            "dataset_key": experiment.dataset_key,
            "groups": list(experiment.groups),
            "description": experiment.description,
            "sensitivity_only": experiment.sensitivity_only,
        }
        for experiment in EXPERIMENTS
        if (
            dataset_key is None
            or experiment.dataset_key == dataset_key
        )
    ]


# =========================================================
# 4. Leakage-aware splitting
# =========================================================

def assert_safe_features(
    X: pd.DataFrame,
    y: pd.Series,
    target_name: Optional[str] = None,
) -> None:
    """
    Basic checks before a supervised split.

    These checks cannot detect every proxy for a target, so label
    provenance and feature semantics must also be reviewed.
    """

    if len(X) != len(y):
        raise ValueError(
            f"X has {len(X)} rows but y has {len(y)} rows."
        )

    if not X.index.equals(y.index):
        raise ValueError(
            "X and y indices do not match. Align them before splitting."
        )

    if target_name is not None and target_name in X.columns:
        raise ValueError(
            f"Target column {target_name!r} is present in X."
        )

    identifiers = [
        column for column in X.columns
        if _is_identifier(column)
    ]

    if identifiers:
        raise ValueError(
            f"Identifier columns must be removed before training: "
            f"{identifiers}"
        )

    if y.isna().any():
        raise ValueError("Target contains missing values.")


def make_holdout_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    groups: Optional[pd.Series] = None,
) -> dict:
    """
    Split raw features into train/test partitions.

    Without groups, use stratified random splitting where possible.
    With groups, keep each group entirely in one partition.

    Group-aware splitting is preferable when multiple rows belong
    to the same person, school, institution, or other shared entity.
    It does not guarantee class stratification.
    """

    assert_safe_features(X, y)

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    if groups is not None:
        if len(groups) != len(X):
            raise ValueError(
                "groups must have the same number of rows as X."
            )

        if groups.isna().any():
            raise ValueError(
                "Group identifiers contain missing values."
            )

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state,
        )

        train_indices, test_indices = next(
            splitter.split(X, y, groups=groups)
        )

        train_groups = set(groups.iloc[train_indices])
        test_groups = set(groups.iloc[test_indices])

        if train_groups.intersection(test_groups):
            raise AssertionError(
                "Group leakage detected across train and test."
            )

        split_method = "GroupShuffleSplit"

    else:
        class_counts = y.value_counts()

        if len(class_counts) < 2:
            raise ValueError(
                "At least two target classes are required."
            )

        stratify = (
            y if class_counts.min() >= 2 else None
        )

        if stratify is None:
            warnings.warn(
                "At least one class has fewer than two observations; "
                "using a non-stratified split.",
                UserWarning,
            )

        train_indices, test_indices = train_test_split(
            np.arange(len(X)),
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

        split_method = (
            "Stratified train_test_split"
            if stratify is not None
            else "Non-stratified train_test_split"
        )

    X_train = X.iloc[train_indices].copy()
    X_test = X.iloc[test_indices].copy()
    y_train = y.iloc[train_indices].copy()
    y_test = y.iloc[test_indices].copy()

    # Check that row partitions do not overlap.
    if set(X_train.index).intersection(set(X_test.index)):
        raise AssertionError("Train/test row overlap detected.")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "train_indices": train_indices,
        "test_indices": test_indices,
        "split_method": split_method,
        "random_state": random_state,
        "test_size": test_size,
    }


# =========================================================
# 5. Build a feature-group experiment without leakage
# =========================================================

def prepare_experiment(
    X: pd.DataFrame,
    y: pd.Series,
    dataset_key: str,
    experiment_name: str,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    groups: Optional[pd.Series] = None,
    allow_sensitivity: bool = False,
) -> dict:
    """
    Select a predefined feature group and create a raw holdout split.

    Fit preprocessing only after this function returns:
        preprocessor.fit_transform(result["X_train"])
        preprocessor.transform(result["X_test"])

    The caller must pass X with the target already removed.
    """

    experiment = get_experiment(
        experiment_name,
        allow_sensitivity=allow_sensitivity,
    )

    if experiment.dataset_key != dataset_key:
        raise ValueError(
            f"Experiment {experiment.name!r} is designed for "
            f"{experiment.dataset_key!r}, not {dataset_key!r}."
        )

    X_selected = select_features(
        X=X,
        dataset_key=dataset_key,
        group_names=experiment.groups,
    )

    assert_safe_features(X_selected, y)

    result = make_holdout_split(
        X=X_selected,
        y=y,
        test_size=test_size,
        random_state=random_state,
        groups=groups,
    )

    result.update({
        "dataset_key": dataset_key,
        "experiment_name": experiment.name,
        "feature_groups": list(experiment.groups),
        "feature_columns": X_selected.columns.tolist(),
        "description": experiment.description,
        "sensitivity_only": experiment.sensitivity_only,
    })

    return result


# =========================================================
# 6. Experimental protocol summary
# =========================================================

def get_split_protocol() -> dict:
    """
    Return the recommended experimental protocol for documentation.
    """

    return {
        "primary_split": {
            "method": "Stratified 80/20 holdout",
            "random_state": RANDOM_STATE,
            "rule": (
                "Split raw rows before fitting preprocessing, "
                "feature selection, or resampling."
            ),
        },
        "grouped_data": {
            "method": "GroupShuffleSplit or grouped cross-validation",
            "rule": (
                "Keep records from the same participant or institution "
                "in only one partition whenever repeated/grouped "
                "observations are possible."
            ),
        },
        "preprocessing": {
            "rule": (
                "Fit imputation, encoding, scaling, feature selection, "
                "and any resampling on training data only."
            ),
        },
        "model_selection": {
            "rule": (
                "Use cross-validation inside the training set for "
                "hyperparameter selection; evaluate the test set once "
                "after the final model is selected."
            ),
        },
        "cross_dataset": {
            "rule": (
                "Do not concatenate the datasets as if they shared "
                "one target. Cross-dataset testing requires a documented "
                "common outcome definition and compatible label mapping."
            ),
        },
        "metrics": [
            "macro_f1",
            "balanced_accuracy",
            "per_class_precision",
            "per_class_recall",
            "confusion_matrix",
        ],
    }


if __name__ == "__main__":
    print("Available experiments:\n")

    for experiment in list_experiments():
        print(
            f"- {experiment['name']} "
            f"({experiment['dataset_key']})"
        )
        print(f"  Groups: {experiment['groups']}")
        print(
            f"  Sensitivity only: "
            f"{experiment['sensitivity_only']}"
        )

    print("\nRecommended split protocol:")
    print(get_split_protocol())
