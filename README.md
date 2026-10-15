# Top-K Career Prediction System

### An Explainable, Multi-Source Career Recommendation Framework

A research-oriented machine-learning project exploring **Top-K career
recommendations** from academic performance, technical and transferable
skills, personality-related responses, and RIASEC career-interest
information. The intended workflow compares feature groups and
predictive models, evaluates classification and ranking quality, and
explains model outputs where supported.

> **Research caution:** This is a research prototype, not a validated
> career-counselling or employment-outcome system. Dataset provenance
> and label-generation rules must be established before making strong
> claims. Report only experiments that have actually been run.

## Contents

-   [Objectives](#objectives)
-   [Datasets](#datasets)
-   [Research and data-quality
    considerations](#research-and-data-quality-considerations)
-   [Workflow](#workflow)
-   [Repository structure](#repository-structure)
-   [Setup](#setup)
-   [Running the project](#running-the-project)
-   [Evaluation](#evaluation)
-   [Explainability and responsible
    use](#explainability-and-responsible-use)
-   [Implementation status](#implementation-status)
-   [Troubleshooting](#troubleshooting)
-   [Reproducibility](#reproducibility)
-   [License and data attribution](#license-and-data-attribution)

## Objectives

1.  Audit each dataset independently and document its source and target
    definition.
2.  Prevent target leakage and remove direct identifiers from model
    features.
3.  Compare baseline models with reproducible preprocessing and data
    splits.
4.  Study academic/skills, personality-related, and RIASEC feature
    groups.
5.  Develop ranked Top-K recommendations and evaluate them with suitable
    metrics.
6.  Apply explainability methods, such as SHAP where compatible with the
    model.
7.  Report limitations, privacy risks, and generalisation constraints.

## Datasets

Put authorised raw files in `data/raw/`. The configuration has used
these filenames:

  -----------------------------------------------------------------------------------------------
  Dataset           Expected filename                       Known structure    Intended role
  ----------------- --------------------------------------- ------------------ ------------------
  Student           `Student_Personality_Original.csv`      About 1,863 rows × Explore prediction
  Personality                                               18 columns;        of stated
                                                            demographics,      preference using
                                                            questionnaire      permitted,
                                                            fields, and a      non-identifying
                                                            self-reported      features
                                                            career-interest    
                                                            response           

  Career Data       `career_data.csv`                       2,400 rows × 12    Baseline
                                                            columns;           classification
                                                            academic/skill     using
                                                            fields, RIASEC     academic/skills
                                                            scores, and        features;
                                                            `Career` target    RIASEC-inclusive
                                                            with six classes   experiments only
                                                                               as explicit
                                                                               sensitivity
                                                                               analyses

  Career Interest   `DataSet Career Interest Trends.xlsx`   About 981          Data-quality and
  Trends                                                    participant        descriptive
                                                            records, 39 RIASEC analysis;
                                                            questionnaire      supervised
                                                            items, derived     prediction only if
                                                            totals/averages,   the target is
                                                            demographics, and  independent of the
                                                            a recorded RIASEC  input features
                                                            code               
  -----------------------------------------------------------------------------------------------

These are previously inspected dataset descriptions. Verify row counts,
columns, worksheet layout, filenames, and paths against the actual files
before publishing results.

### Do not merge datasets without justification

These files do not automatically represent the same participants or
share a common target definition. Do not concatenate or row-merge them
simply because all concern careers. Run separate experiments unless
participant-level linkage and a defensible target mapping are
documented. If comparing results across datasets, explain their
different populations, features, and labels.

## Research and data-quality considerations

-   **Target meaning:** The Student Personality target is a
    self-reported preference, not a verified employment outcome.
-   **Circularity:** The Career Interest Trends RIASEC code may be
    derived from the same questionnaire responses or summary scores.
    Predicting it from those same items may only reconstruct a scoring
    rule.
-   **Label provenance:** Establish how `Career` labels in
    `career_data.csv` were generated. If labels were assigned using
    included variables, scores may reflect that construction process.
-   **Privacy:** Exclude names, email addresses, direct IDs, and other
    identifiers from model features. Do not expose raw participant
    records in a dashboard or publish private data.
-   **Questionnaire validity:** Do not interpret numbered questions as
    validated Big Five/personality scores unless the questionnaire,
    scoring procedure, and source support this.
-   **Leakage prevention:** Fit imputers, scalers, encoders, feature
    selection, and resampling on training data only; transform test data
    using the fitted objects.
-   **Generalisation:** Report per-class performance and sample sizes.
    Avoid broad claims from small or convenience samples.

## Workflow

1.  **Ingest:** Load CSV/Excel files and check paths, schemas, target
    columns, and sheet structure.
2.  **Audit:** Check missing values, duplicates, unexpected labels,
    class counts, and identifiers.
3.  **Prepare:** Normalize labels and columns; separate target from
    features; remove direct identifiers.
4.  **Select features:** Define feature groups for each experiment and
    clearly label RIASEC-inclusive sensitivity runs.
5.  **Split:** Create reproducible train/test partitions, stratified
    when appropriate.
6.  **Preprocess:** Impute, scale numeric features where useful, and
    encode categorical features within a pipeline.
7.  **Train and compare:** Evaluate baselines under consistent split and
    feature definitions.
8.  **Rank:** Convert valid model scores into ordered recommendations
    and evaluate Top-K quality.
9.  **Explain:** Apply suitable feature-importance or SHAP methods and
    document their limitations.
10. **Report:** Save metrics, plots, run configurations, and
    limitations.

## Repository structure

The repository is organised around these core files. Items marked
planned/ongoing should not be understood as complete until implemented
and tested.

``` text
Top-K_Career_Prediction_Sytem/
├── data/
│   ├── raw/                   # Source data; check permissions before committing
│   └── processed/             # Generated data; normally not committed
├── src/
│   ├── config.py              # Dataset metadata, paths, output directories
│   ├── data_loader.py         # Loading and schema audit
│   ├── preprocessing.py       # Target/feature preparation and preprocessing
│   ├── feature_groups.py      # Feature-group selection
│   ├── train_models.py        # Planned/ongoing: model training
│   ├── evaluate_models.py     # Planned/ongoing: evaluation
│   ├── explainability.py      # Planned/ongoing: model explanations
│   └── recommendation.py      # Planned/ongoing: Top-K ranking
├── tests/                     # Automated tests
├── models/                    # Generated models; normally not committed
├── results/                   # Metrics, plots, reports
├── run_experiments.py         # Current preprocessing smoke-test entry point
├── requirements.txt
├── .gitignore
└── README.md
```

Update this tree when modules are added or renamed.

## Setup

### Requirements

-   Python 3.10 or a version supported by the installed packages
-   `pip` and Git
-   Dataset files configured in `src/config.py`

### 1. Clone the repository

``` bash
git clone https://github.com/abhikrm0102/Top-K_Career_Prediction_Sytem.git
cd Top-K_Career_Prediction_Sytem
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

``` powershell
python -m venv env
.\env\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation, use Command Prompt:

``` bat
env\Scripts\activate.bat
```

Or use the environment's Python directly from PowerShell:

``` powershell
.\env\Scripts\python.exe -m pip install --upgrade pip
```

**macOS/Linux**

``` bash
python3 -m venv env
source env/bin/activate
python -m pip install --upgrade pip
```

### 3. Install dependencies

``` bash
python -m pip install -r requirements.txt
python -m pip check
```

Keep `requirements.txt` limited to packages actually used by the
project; unnecessary dependencies increase installation time and may
cause compatibility conflicts.

### 4. Add the datasets

Put authorised files in `data/raw/`. Match their actual filenames to
`src/config.py`; update the configuration if the source filenames
differ. Do not rename files blindly.

### 5. Check configuration and loading

Run from the repository root:

``` bash
python -m src.config
python -m src.data_loader
```

These commands depend on the module entry points being present and
functional in the current checkout.

## Running the project

### Current preprocessing experiment

``` bash
python run_experiments.py
```

In the reviewed version, this script loads `career_data`, prepares a
train/test split, fits preprocessing on the training data, transforms
the test data, and prints experiment/feature/shape information. **It is
a preprocessing smoke test, not a completed model-training or Top-K
evaluation run.**

### Tests

``` bash
python -m pytest tests -v
```

Record the actual output and resolve failures before claiming the tests
pass.

Document the exact commands for model training, ranking, or a dashboard
only after those entry points exist and have been tested.

## Evaluation

Candidate classification baselines include Logistic Regression,
K-Nearest Neighbours (KNN), Support Vector Machine (SVM), Random Forest,
and XGBoost when installed and appropriate. Add a neural network only if
it is justified by the data size and research question. Tune
hyperparameters using training data or cross-validation, not the final
test set.

### Classification metrics

As appropriate, report accuracy, macro-F1, weighted-F1, per-class
precision and recall, balanced accuracy, and a confusion matrix. Use
ROC-AUC only when the class setup and model outputs make it appropriate.

### Top-K metrics

Define relevance before evaluating rankings. Suitable measures may
include: - **Hit Rate@K:** whether a relevant career appears in the
first K results. - **Precision@K:** proportion of the first K
recommendations considered relevant. - **Recall@K:** proportion of
relevant options retrieved among the first K. - **NDCG@K:** ranking
quality when graded relevance is available.

For single-label datasets, document exactly how ranking metrics are
defined. Ordinary classification accuracy is not a Top-K metric. Compare
multiple K values where appropriate and report variability across splits
or cross-validation.

## Explainability and responsible use

Explainability can help inspect a model, but it does not establish
causation or prove that a recommendation is right for an individual. If
using SHAP, document the model, dataset, background sample, and any
approximation. Distinguish global feature importance from local
explanations, and be careful with correlated or derived questionnaire
variables.

Treat the system as a research prototype or decision-support aid, not
the sole basis for education, employment, or career decisions.
Recommendations should be considered alongside an individual's
interests, goals, experience, and advice from qualified counsellors.

## Implementation status

Based on the latest source review, the project includes dataset
configuration, data-loading and preprocessing modules, feature-group
selection, and a script that exercises preprocessing on `career_data`.

Before claiming a complete end-to-end system, verify or complete: - \[
\] Configured dataset paths, worksheet layouts, and target columns match
the actual files. - \[ \] Questionnaire-item detection handles names
such as `item_R_01` and `item_I_02`. - \[ \] Publicly tracked files
contain no unintended personal data or secrets. - \[ \] Automated tests
run successfully. - \[ \] Model training and evaluation are implemented
and validated. - \[ \] Top-K ranking and ranking metrics are implemented
and validated. - \[ \] Explanations are generated only for supported,
trained models. - \[ \] Run configurations and results are saved. - \[
\] Dataset provenance, licensing, and label-generation rules are
documented. - \[ \] Any dashboard is tested end to end.

Update this checklist as work is completed.

## Troubleshooting

### `FileNotFoundError`

Run commands from the repository root, check the exact
filename/extension under `data/raw/`, and update `src/config.py` if
necessary.

### `ImportError` or `KeyError`

Check that dataset keys and configuration fields expected by
`data_loader.py` match `src/config.py`, and verify target-column
spelling against the actual data.

### PowerShell cannot activate `env`

Use `env\Scripts\activate.bat` in Command Prompt or call
`env\Scripts\python.exe` directly. Avoid changing machine-wide execution
policy just to activate a project environment.

### Unexpectedly high scores

Investigate target leakage, duplicates or overlapping records across
splits, features derived from the target, and questionnaire variables
that directly determine the label. High scores do not by themselves
establish real-world predictive validity.

## Reproducibility checklist

Record for each experiment: - Dataset source, filename, version, and
licence - Target definition and label-generation process -
Included/excluded feature groups - Split strategy, random seed, and
sample counts - Preprocessing steps, model parameters, and software
versions - Classification and Top-K metrics, where applicable -
Per-class results and subgroup checks where justified - Limitations and
leakage risks

## Contributing

Focused contributions and pull requests are welcome. 1. Keep dataset
assumptions explicit. 2. Add or update tests when preprocessing or
feature selection changes. 3. Do not commit virtual environments,
credentials, private participant data, or generated artefacts
unnecessarily. 4. Include commands and outputs sufficient to reproduce
reported results.

## License and data attribution

Do not assume a code or dataset licence from the repository name or file
format. Before redistributing code or data, add the appropriate project
licence and document each dataset's source, permission, licence terms,
and required attribution. If raw data cannot be redistributed, explain
how to obtain it lawfully instead of committing it.

------------------------------------------------------------------------

**Repository:**
https://github.com/abhikrm0102/Top-K_Career_Prediction_Sytem
