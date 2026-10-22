"""Data loading, validation, and train-test splitting for EduPredict AI.

Handles dataset ingestion, schema validation, data integrity audits,
target variable derivation, and strictly prevents data leakage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Complete column specifications for UCI Student Performance dataset
ALL_EXPECTED_COLUMNS: List[str] = [
    "school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
    "Mjob", "Fjob", "reason", "guardian", "traveltime", "studytime", "failures",
    "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet",
    "romantic", "famrel", "freetime", "goout", "Dalc", "Walc", "health",
    "absences", "G1", "G2", "G3"
]

# Demographics, behavioral, and academic background features available prior to term exams
EARLY_FEATURES: List[str] = [
    "school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
    "Mjob", "Fjob", "reason", "guardian", "traveltime", "studytime", "failures",
    "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet",
    "romantic", "famrel", "freetime", "goout", "Dalc", "Walc", "health",
    "absences"
]

# Features when mid-term grades G1 and G2 are available
MIDTERM_FEATURES: List[str] = EARLY_FEATURES + ["G1", "G2"]

CATEGORICAL_FEATURES: List[str] = [
    "school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob",
    "reason", "guardian", "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic"
]

NUMERIC_FEATURES_EARLY: List[str] = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences"
]

VALID_CATEGORICAL_VALUES: Dict[str, List[str]] = {
    "school": ["GP", "MS"],
    "sex": ["F", "M"],
    "address": ["U", "R"],
    "famsize": ["LE3", "GT3"],
    "Pstatus": ["T", "A"],
    "Mjob": ["teacher", "health", "services", "at_home", "other"],
    "Fjob": ["teacher", "health", "services", "at_home", "other"],
    "reason": ["home", "reputation", "course", "other"],
    "guardian": ["mother", "father", "other"],
    "schoolsup": ["yes", "no"],
    "famsup": ["yes", "no"],
    "paid": ["yes", "no"],
    "activities": ["yes", "no"],
    "nursery": ["yes", "no"],
    "higher": ["yes", "no"],
    "internet": ["yes", "no"],
    "romantic": ["yes", "no"],
}

NUMERIC_RANGES: Dict[str, Tuple[float, float]] = {
    "age": (15, 25),
    "Medu": (0, 4),
    "Fedu": (0, 4),
    "traveltime": (1, 4),
    "studytime": (1, 4),
    "failures": (0, 4),
    "famrel": (1, 5),
    "freetime": (1, 5),
    "goout": (1, 5),
    "Dalc": (1, 5),
    "Walc": (1, 5),
    "health": (1, 5),
    "absences": (0, 100),
    "G1": (0, 20),
    "G2": (0, 20),
    "G3": (0, 20),
}


@dataclass
class DatasetBundle:
    """Encapsulates dataset partitions, target definitions, and metadata."""
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_reg_train: pd.Series
    y_reg_test: pd.Series
    y_clf_train: pd.Series
    y_clf_test: pd.Series
    feature_names: List[str]
    categorical_cols: List[str]
    numeric_cols: List[str]
    scenario: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def locate_dataset_file(filepath: Optional[str | Path] = None) -> Path:
    """Resolves dataset location across workspace directories."""
    if filepath is not None:
        p = Path(filepath)
        if p.exists():
            return p
        raise FileNotFoundError(f"Requested dataset file not found at: {filepath}")

    # Standard locations to probe in order
    candidate_paths = [
        Path("data/student-mat.csv"),
        Path("AIML PROJECT 2/data/student-mat.csv"),
        Path("../student/student-mat.csv"),
        Path("d:/7th Sem/Artifical Intelligence and Machine Learning/AIML PROJECT 2/data/student-mat.csv"),
        Path("d:/7th Sem/Artifical Intelligence and Machine Learning/student/student-mat.csv"),
    ]

    for cand in candidate_paths:
        if cand.exists():
            return cand.resolve()

    raise FileNotFoundError("Could not find student-mat.csv in workspace. Checked paths: "
                            f"{[str(c) for c in candidate_paths]}")


def load_raw_dataset(filepath: Optional[str | Path] = None, sep: str = ";") -> pd.DataFrame:
    """Loads student performance dataset from CSV, preserving raw data integrity."""
    resolved_path = locate_dataset_file(filepath)
    logger.info(f"Loading student performance dataset from: {resolved_path}")
    df = pd.read_csv(resolved_path, sep=sep)
    return df


def validate_dataset(df: pd.DataFrame, include_prior_grades: bool = False) -> Dict[str, Any]:
    """Validates schema, null values, duplicates, and range constraints."""
    missing_cols = [c for c in ALL_EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    null_count = int(df.isnull().sum().sum())
    dup_count = int(df.duplicated().sum())
    total_records = len(df)

    # Validate value bounds for critical columns
    range_violations = {}
    for col, (min_val, max_val) in NUMERIC_RANGES.items():
        if col in df.columns:
            out_of_bounds = df[(df[col] < min_val) | (df[col] > max_val)]
            if len(out_of_bounds) > 0:
                range_violations[col] = len(out_of_bounds)

    summary = {
        "total_records": total_records,
        "total_columns": len(df.columns),
        "missing_values": null_count,
        "duplicate_records": dup_count,
        "range_violations": range_violations,
        "is_valid": null_count == 0 and dup_count == 0 and len(range_violations) == 0,
    }
    return summary


def prepare_dataset_bundle(
    df: Optional[pd.DataFrame] = None,
    filepath: Optional[str | Path] = None,
    include_prior_grades: bool = False,
    test_size: float = 0.20,
    random_state: int = 42,
) -> DatasetBundle:
    """Prepares stratified train and test partitions strictly preventing data leakage.

    In the default early prediction scenario (include_prior_grades=False),
    intermediate grades G1 and G2 are excluded alongside G3, allowing early-term
    prediction before any period examinations take place.

    Target definitions:
      - Regression target: G3 (final grade on 0-20 scale)
      - Classification target: academic_risk = 1 if G3 < 10 else 0 (10/20 is passing standard)
    """
    if df is None:
        df = load_raw_dataset(filepath)

    validation_summary = validate_dataset(df, include_prior_grades=include_prior_grades)
    logger.info(f"Dataset validation passed: {validation_summary['total_records']} rows, "
                f"{validation_summary['missing_values']} missing, {validation_summary['duplicate_records']} duplicates.")

    # Target variables definition
    y_reg = df["G3"].astype(float)
    # Academic risk: 1 = At Risk (Fail: G3 < 10), 0 = Safe (Pass: G3 >= 10)
    y_clf = (y_reg < 10.0).astype(int)

    # Determine input feature subspace
    if include_prior_grades:
        scenario = "midterm_prediction"
        feature_cols = [c for c in MIDTERM_FEATURES if c in df.columns]
        num_cols = NUMERIC_FEATURES_EARLY + ["G1", "G2"]
    else:
        scenario = "early_prediction"
        feature_cols = [c for c in EARLY_FEATURES if c in df.columns]
        num_cols = [c for c in NUMERIC_FEATURES_EARLY if c in df.columns]

    cat_cols = [c for c in CATEGORICAL_FEATURES if c in feature_cols]

    # Explicit leakage prevention check: target must NEVER be in X
    assert "G3" not in feature_cols, "CRITICAL ERROR: Target G3 is present in feature set!"
    if not include_prior_grades:
        assert "G1" not in feature_cols and "G2" not in feature_cols, (
            "CRITICAL ERROR: G1/G2 present in early prediction feature set!"
        )

    X = df[feature_cols].copy()

    # Stratified train/test split based on binary risk to preserve class balance
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X,
        y_reg,
        y_clf,
        test_size=test_size,
        random_state=random_state,
        stratify=y_clf,
    )

    metadata = {
        "scenario": scenario,
        "n_samples": len(df),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": len(feature_cols),
        "categorical_count": len(cat_cols),
        "numeric_count": len(num_cols),
        "test_size": test_size,
        "random_state": random_state,
        "target_reg_mean": float(y_reg.mean()),
        "target_reg_std": float(y_reg.std()),
        "risk_rate_overall": float(y_clf.mean()),
        "risk_rate_train": float(y_clf_train.mean()),
        "risk_rate_test": float(y_clf_test.mean()),
        "validation_summary": validation_summary,
    }

    return DatasetBundle(
        X_train=X_train,
        X_test=X_test,
        y_reg_train=y_reg_train,
        y_reg_test=y_reg_test,
        y_clf_train=y_clf_train,
        y_clf_test=y_clf_test,
        feature_names=feature_cols,
        categorical_cols=cat_cols,
        numeric_cols=num_cols,
        scenario=scenario,
        metadata=metadata,
    )


if __name__ == "__main__":
    bundle = prepare_dataset_bundle()
    print("Dataset Bundle Prepared Successfully:")
    print(f"Scenario: {bundle.scenario}")
    print(f"Train samples: {bundle.metadata['n_train']}, Test samples: {bundle.metadata['n_test']}")
    print(f"Features ({len(bundle.feature_names)}): {bundle.feature_names}")
    print(f"Overall Risk Rate: {bundle.metadata['risk_rate_overall']:.1%}")
    print(f"Train Risk Rate: {bundle.metadata['risk_rate_train']:.1%}")
    print(f"Test Risk Rate: {bundle.metadata['risk_rate_test']:.1%}")
