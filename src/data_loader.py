
"""
data_loader.py
---------------
Run code: -
.\.venv\Scripts\python.exe -m src.data_loader
---------------
> results/reports/schema_audit.json — detailed schema, missing-value, and target-label report.
> results/reports/schema_audit.csv — column-by-column audit that you can open in Excel.
---------------
Loads and audits the three career-recommendation datasets.

Responsibilities:
1. Load CSV and Excel files.
2. Normalize column names and selected text labels.
3. Parse the RIASEC questionnaire workbook.
4. Inspect schemas, missing values, duplicates, and target labels.
5. Save audit reports as CSV and JSON.

This module does not merge datasets or impute missing values.
"""

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd

try:
    from .config import DATASETS, REPORTS_DIR, create_directories
except ImportError:
    from config import DATASETS, REPORTS_DIR, create_directories


RIASEC_CODES = ("R", "I", "A", "S", "E", "C")


# ============================================================
# 1. COMMON UTILITIES
# ============================================================

def normalize_text(value):
    """Trim surrounding whitespace and collapse repeated spaces."""
    if pd.isna(value):
        return value

    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_column_names(df):
    """Remove accidental whitespace from column names."""
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def check_file_exists(path):
    """Raise a useful error if a dataset file is missing."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found:\n{path}\n\n"
            "Place the original file inside data/raw/ "
            "and check its filename in src/config.py."
        )

    return path


# ============================================================
# 2. DATASET 1: STUDENT PERSONALITY
# ============================================================

def load_student_personality(path):
    """
    Load the student personality dataset.

    Target:
        Which of the following career fields are you most interested in?

    The target represents self-reported career preference, not a
    verified employment outcome.
    """
    path = check_file_exists(path)

    df = pd.read_csv(path)
    df = normalize_column_names(df)

    target = DATASETS["student_personality"]["target"]

    if target not in df.columns:
        raise ValueError(
            f"Expected target column {target!r} was not found.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    df[target] = df[target].map(normalize_text)

    return df


# ============================================================
# 3. DATASET 2: CAREER PREDICTION
# ============================================================

def load_career_data(path):
    """
    Load the academic, technical-skills, RIASEC, and career-label
    dataset.

    Target:
        Career

    Target-generation provenance should be verified before
    interpreting model performance as genuine career prediction.
    """
    path = check_file_exists(path)

    df = pd.read_csv(path)
    df = normalize_column_names(df)

    target = DATASETS["career_data"]["target"]

    if target not in df.columns:
        raise ValueError(
            f"Expected target column {target!r} was not found.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    df[target] = df[target].map(normalize_text)

    return df


# ============================================================
# 4. DATASET 3: RIASEC EXCEL WORKBOOK
# ============================================================

def load_riasec_workbook(path, sheet_name="Data"):
    """
    Convert the workbook's Data sheet into a standard DataFrame.

    Workbook layout observed in the supplied file:
        Row 1: section/header information
        Row 2: item labels
        Row 3 onward: participant records

    The 39 questionnaire items occupy zero-based columns 3:42.
    The six recorded RIASEC totals occupy columns 43:49.
    The six recorded RIASEC averages occupy columns 50:56.
    The recorded career-interest code is stored across columns 57:63.

    Important:
        The recorded career-interest code is derived from the
        questionnaire/profile. Predicting it from those same items
        would create target leakage.
    """
    path = check_file_exists(path)

    raw = pd.read_excel(
        path,
        sheet_name=sheet_name,
        header=None
    )

    if raw.shape[1] < 63:
        raise ValueError(
            f"Expected at least 63 columns in sheet {sheet_name!r}; "
            f"found {raw.shape[1]}."
        )

    if len(raw) < 3:
        raise ValueError(
            "The workbook does not contain the expected header "
            "rows and participant records."
        )

    records = raw.iloc[2:].copy().reset_index(drop=True)

    # Participant metadata
    output = pd.DataFrame()

    output["participant_id"] = pd.to_numeric(
        records.iloc[:, 0],
        errors="coerce"
    )

    output["sex"] = records.iloc[:, 1].map(normalize_text)

    output["school"] = pd.to_numeric(
        records.iloc[:, 2],
        errors="coerce"
    )

    # Questionnaire responses: 39 columns
    items = records.iloc[:, 3:42].apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Build stable item names using the second header row.
    item_names = []

    for index, header in enumerate(raw.iloc[1, 3:42], start=1):
        header = normalize_text(header)

        if pd.notna(header):
            match = re.match(
                r"([RIASEC])[-_ ]?(\d+)",
                str(header),
                flags=re.IGNORECASE
            )
        else:
            match = None

        if match:
            name = (
                f"item_{match.group(1).upper()}_"
                f"{match.group(2)}"
            )
        else:
            name = f"item_{index:02d}"

        item_names.append(name)

    items.columns = item_names

    output = pd.concat(
        [output, items],
        axis=1
    )

    # Recorded RIASEC totals
    for code, column_index in zip(
        RIASEC_CODES,
        range(43, 49)
    ):
        output[f"{code}_total_recorded"] = pd.to_numeric(
            records.iloc[:, column_index],
            errors="coerce"
        )

    # Recorded RIASEC averages
    for code, column_index in zip(
        RIASEC_CODES,
        range(50, 56)
    ):
        output[f"{code}_average_recorded"] = pd.to_numeric(
            records.iloc[:, column_index],
            errors="coerce"
        )

    # Extract one recorded RIASEC code per participant.
    target_cells = records.iloc[:, 57:63]

    def extract_interest_code(row):
        values = [
            str(value).strip().upper()
            for value in row
            if pd.notna(value)
        ]

        codes = [
            value for value in values
            if value in RIASEC_CODES
        ]

        if not codes:
            return np.nan

        # Normally exactly one category is populated.
        if len(set(codes)) > 1:
            return np.nan

        return codes[0]

    output["career_interest_code"] = target_cells.apply(
        extract_interest_code,
        axis=1
    )

    # Remove blank spreadsheet rows without imputing questionnaire data.
    output = output.loc[
        output["participant_id"].notna()
    ].reset_index(drop=True)

    return output


# ============================================================
# 5. UNIFIED DATASET LOADER
# ============================================================

def load_dataset(dataset_key):
    """
    Load a dataset by its key in config.py.

    Valid keys:
        student_personality
        career_data
        career_interest_trends
    """
    if dataset_key not in DATASETS:
        raise KeyError(
            f"Unknown dataset: {dataset_key!r}\n"
            f"Valid keys: {list(DATASETS.keys())}"
        )

    config = DATASETS[dataset_key]
    path = config["path"]
    file_format = config["format"]

    if file_format == "csv":
        if dataset_key == "student_personality":
            return load_student_personality(path)

        if dataset_key == "career_data":
            return load_career_data(path)

    elif file_format == "xlsx_riasec":
        return load_riasec_workbook(path)

    raise ValueError(
        f"Unsupported format {file_format!r} "
        f"for dataset {dataset_key!r}."
    )


# ============================================================
# 6. DATA QUALITY AND SCHEMA AUDIT
# ============================================================

def audit_dataframe(df, dataset_key):
    """Return a structured report about the dataset."""
    config = DATASETS[dataset_key]
    target = config["target"]

    missing_counts = df.isna().sum()

    numeric_ranges = {}

    for column in df.select_dtypes(include="number").columns:
        values = df[column].dropna()

        if not values.empty:
            numeric_ranges[column] = {
                "minimum": float(values.min()),
                "maximum": float(values.max())
            }

    if target in df.columns:
        target_counts = {
            str(label): int(count)
            for label, count in
            df[target].value_counts(dropna=False).items()
        }

        target_missing = int(df[target].isna().sum())
    else:
        target_counts = None
        target_missing = None

    report = {
        "dataset_key": dataset_key,
        "dataset_name": config["path"].name,
        "configured_path": str(config["path"]),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),

        "data_types": {
            column: str(dtype)
            for column, dtype in df.dtypes.items()
        },

        "missing_count_by_column": {
            column: int(count)
            for column, count in missing_counts.items()
        },

        "missing_percentage_by_column": {
            column: round(
                float(count / max(len(df), 1) * 100),
                3
            )
            for column, count in missing_counts.items()
        },

        "total_missing_cells": int(missing_counts.sum()),

        "duplicate_full_rows": int(
            df.duplicated().sum()
        ),

        "target_column": (
            target if target in df.columns else None
        ),

        "target_missing": target_missing,
        "target_class_counts": target_counts,
        "numeric_ranges": numeric_ranges,

        "task_note": config.get("task_note", ""),
    }

    return report


# ============================================================
# 7. AUDIT ALL THREE DATASETS
# ============================================================

def audit_all_datasets(save=True):
    """
    Load and audit each dataset independently.

    Saves:
        results/reports/schema_audit.json
        results/reports/schema_audit.csv
    """
    create_directories()

    audits = {}

    for dataset_key in DATASETS:
        try:
            df = load_dataset(dataset_key)

            audits[dataset_key] = audit_dataframe(
                df,
                dataset_key
            )

        except Exception as error:
            audits[dataset_key] = {
                "dataset_key": dataset_key,
                "configured_path": str(
                    DATASETS[dataset_key]["path"]
                ),
                "error": str(error),
            }

    if save:
        REPORTS_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        json_path = REPORTS_DIR / "schema_audit.json"

        with open(
            json_path,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                audits,
                file,
                indent=4,
                ensure_ascii=False
            )

        rows = []

        for dataset_key, audit in audits.items():
            if "error" in audit:
                rows.append({
                    "dataset": dataset_key,
                    "error": audit["error"],
                })
                continue

            for column in audit["column_names"]:
                rows.append({
                    "dataset": dataset_key,
                    "rows": audit["rows"],
                    "columns": audit["columns"],
                    "column": column,
                    "dtype": audit["data_types"].get(column),
                    "missing_count": audit[
                        "missing_count_by_column"
                    ].get(column),
                    "missing_percentage": audit[
                        "missing_percentage_by_column"
                    ].get(column),
                    "target": audit["target_column"],
                })

        csv_path = REPORTS_DIR / "schema_audit.csv"

        pd.DataFrame(rows).to_csv(
            csv_path,
            index=False
        )

        print(f"\nJSON audit saved to: {json_path}")
        print(f"CSV audit saved to:  {csv_path}")

    return audits


# ============================================================
# 8. COMMAND-LINE ENTRY POINT
# ============================================================

if __name__ == "__main__":
    audits = audit_all_datasets(save=True)

    print("\nDATASET AUDIT SUMMARY")
    print("=" * 60)

    for dataset_key, audit in audits.items():
        print(f"\nDataset: {dataset_key}")

        if "error" in audit:
            print("Status: ERROR")
            print(audit["error"])
            continue

        print(
            f"Shape: {audit['rows']} rows x "
            f"{audit['columns']} columns"
        )

        print(
            f"Target: {audit['target_column']}"
        )

        print(
            f"Missing cells: "
            f"{audit['total_missing_cells']}"
        )

        print(
            f"Duplicate rows: "
            f"{audit['duplicate_full_rows']}"
        )

        print("Target class counts:")
        print(audit["target_class_counts"])
