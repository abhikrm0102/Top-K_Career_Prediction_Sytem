
"""
config.py
---------
Central configuration for the Explainable Multi-Source
Career Recommendation project.

Project structure expected:
CareerPredictionSystem/
├── data/
│   ├── raw/
│   │   ├── Student_Personality.csv
│   │   ├── career_data.csv
│   │   └── DataSet_Career_Interest_Trends.xlsx
│   └── processed/
├── src/
│   └── config.py
├── models/
│   ├── saved_models/
│   └── preprocessors/
└── results/
    ├── metrics/
    ├── figures/
    └── reports/
"""

from pathlib import Path


# ============================================================
# 1. PROJECT ROOT
# ============================================================

# This file is expected to be inside the src/ directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# 2. DATASET PATHS
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"



DATASETS = {
    "student_personality": {
        "name": "Student Personality Dataset",
        "path": RAW_DATA_DIR / "Student_Personality.csv",
        "format": "csv",
        "target": (
            "Which of the following career fields are you "
            "most interested in?"
        ),
        "task_note": (
            "Predict self-reported career preference, not "
            "verified employment outcomes."
        ),
    },
    "career_data": {
        "name": "Student Career Prediction Dataset",
        "path": RAW_DATA_DIR / "career_data.csv",
        "format": "csv",
        "target": "Career",
        "task_note": (
            "Verify how career labels were generated before "
            "interpreting predictive performance."
        ),
    },
    "career_interest_trends": {
        "name": "Career Interest Trends Dataset",
        "path": RAW_DATA_DIR / "DataSet_Career_Interest_Trends.xlsx",
        "format": "xlsx_riasec",
        "target": "career_interest_code",
        "task_note": (
            "The career-interest code may be derived from "
            "the same RIASEC questionnaire used as input."
        ),
    },
}


# ============================================================
# 3. OUTPUT DIRECTORIES
# ============================================================

MODELS_DIR = PROJECT_ROOT / "models"
SAVED_MODELS_DIR = MODELS_DIR / "saved_models"
PREPROCESSORS_DIR = MODELS_DIR / "preprocessors"

RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORTS_DIR = RESULTS_DIR / "reports"


# ============================================================
# 4. REPRODUCIBILITY AND DATA SPLITTING
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20
VALIDATION_SIZE = 0.10

# Number of folds used for cross-validation.
CV_FOLDS = 5

# Use stratification when the target classes support it.
USE_STRATIFIED_SPLIT = True


# ============================================================
# 5. MACHINE-LEARNING MODELS
# ============================================================

MODEL_NAMES = [
    "Logistic Regression",
    "K-Nearest Neighbors",
    "Decision Tree",
    "Random Forest",
    "Support Vector Machine",
    "XGBoost",
]

# Models requiring special handling or optional dependencies.
OPTIONAL_MODELS = ["XGBoost"]

# Configuration for individual estimators.
MODEL_PARAMS = {
    "Logistic Regression": {
        "max_iter": 2000,
        "class_weight": "balanced",
    },
    "K-Nearest Neighbors": {
        "n_neighbors": 5,
        "weights": "distance",
    },
    "Decision Tree": {
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
    },
    "Random Forest": {
        "n_estimators": 300,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    },
    "Support Vector Machine": {
        "C": 1.0,
        "kernel": "rbf",
        "probability": True,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
    },
    "XGBoost": {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    },
}


# ============================================================
# 6. FEATURE GROUPS
# ============================================================

FEATURE_GROUPS = {
    "academic": [
        # Add verified academic-score column names here.
    ],
    "skills": [
        # Add verified technical/soft-skill column names here.
    ],
    "personality": [
        # Add verified BFI-10 personality column names here.
    ],
    "riasec": [
        # Add verified RIASEC score column names here.
    ],
    "riasec_questionnaire": [
        # Add verified questionnaire-item column names here.
    ],
    "demographics": [
        # Add demographic columns only when justified and appropriate.
    ],
}

# These are feature groups, not assumptions that every dataset
# contains every group. Verify each dataset's schema first.


# ============================================================
# 7. EVALUATION METRICS
# ============================================================

CLASSIFICATION_METRICS = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
]

# Top-k metrics are appropriate only when the task supports
# ranked career-class predictions.
TOP_K_VALUES = [1, 3]

SAVE_CONFUSION_MATRIX = True
SAVE_CLASSIFICATION_REPORT = True


# ============================================================
# 8. ABLATION STUDY
# ============================================================

ABLATION_EXPERIMENTS = {
    "academic_only": ["academic"],
    "skills_only": ["skills"],
    "personality_only": ["personality"],
    "riasec_only": ["riasec"],
    "academic_and_skills": ["academic", "skills"],
    "personality_and_riasec": ["personality", "riasec"],
    "all_available_groups": [
        "academic",
        "skills",
        "personality",
        "riasec",
    ],
}

# Run an ablation only when the required groups exist in the
# selected dataset. Do not combine unrelated rows across datasets.


# ============================================================
# 9. EXPLAINABILITY
# ============================================================

ENABLE_SHAP = True

# Limit expensive explanation calculations on large datasets.
SHAP_MAX_SAMPLES = 200

# Number of features to show in global explanation charts.
TOP_FEATURES_TO_DISPLAY = 15


# ============================================================
# 10. CAREER RECOMMENDATION SETTINGS
# ============================================================

TOP_N_RECOMMENDATIONS = 3

# A recommendation is based on model scores/probabilities.
# Do not describe these scores as calibrated probabilities
# unless calibration has been evaluated.


# ============================================================
# 11. FAIRNESS AND ROBUSTNESS
# ============================================================

ENABLE_FAIRNESS_ANALYSIS = True

# Define appropriate groups only after inspecting the data
# and establishing a valid research justification.
FAIRNESS_GROUP_COLUMNS = []

# Example checks can include subgroup performance,
# class imbalance, missingness, and sensitivity to features.


# ============================================================
# 12. DATA VALIDATION
# ============================================================

DROP_DUPLICATES = True
HANDLE_MISSING_VALUES = True

# Avoid silently deleting records or columns without documenting it.
MAX_MISSING_FEATURE_FRACTION = 0.40

# Target columns must never be included among model input features.
PREVENT_TARGET_LEAKAGE = True


# ============================================================
# 13. STREAMLIT DASHBOARD
# ============================================================

APP_TITLE = "Explainable Career Recommendation System"

APP_ICON = "🎓"

APP_LAYOUT = "wide"


# ============================================================
# 14. LOGGING
# ============================================================

LOG_LEVEL = "INFO"
LOG_FILE = RESULTS_DIR / "project.log"


# ============================================================
# 15. DIRECTORY INITIALIZATION
# ============================================================

def create_project_directories():
    """Create required folders if they do not already exist."""

    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        SAVED_MODELS_DIR,
        PREPROCESSORS_DIR,
        METRICS_DIR,
        FIGURES_DIR,
        REPORTS_DIR,
        LOG_FILE.parent,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# Compatibility alias used by data_loader.py
create_directories = create_project_directories

def create_directories():
    """Compatibility wrapper for data_loader.py."""
    create_project_directories()

def validate_dataset_paths():
    """
    Return a dictionary showing whether each configured dataset
    file exists. This function does not load or modify the data.
    """

    return {
        dataset_key: {
            "path": str(dataset_config["path"]),
            "exists": dataset_config["path"].is_file(),
        }
        for dataset_key, dataset_config in DATASETS.items()
    }


if __name__ == "__main__":
    create_project_directories()

    print("Project root:", PROJECT_ROOT)
    print("\nDataset availability:")

    for dataset_key, status in validate_dataset_paths().items():
        print(
            f"- {dataset_key}: "
            f"{'FOUND' if status['exists'] else 'NOT FOUND'} "
            f"({status['path']})"
        )
