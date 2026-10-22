"""Unit tests for dataset loading, schema validation, and leakage prevention."""

import sys
from pathlib import Path
import pytest
import pandas as pd

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import (
    ALL_EXPECTED_COLUMNS,
    EARLY_FEATURES,
    load_raw_dataset,
    locate_dataset_file,
    prepare_dataset_bundle,
    validate_dataset,
)


def test_locate_dataset_file():
    """Confirms dataset file is located accurately in the workspace."""
    path = locate_dataset_file()
    assert path.exists(), f"Located dataset does not exist: {path}"
    assert path.suffix == ".csv", f"Dataset is not a CSV: {path}"


def test_load_raw_dataset():
    """Confirms dataset is loaded with 395 rows and 33 columns."""
    df = load_raw_dataset()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 395
    assert len(df.columns) == 33
    for col in ALL_EXPECTED_COLUMNS:
        assert col in df.columns, f"Missing expected column: {col}"


def test_validate_dataset_integrity():
    """Verifies dataset has zero nulls and zero duplicates."""
    df = load_raw_dataset()
    summary = validate_dataset(df)
    assert summary["missing_values"] == 0
    assert summary["duplicate_records"] == 0
    assert summary["is_valid"] is True


def test_leakage_prevention_early_prediction():
    """Confirms strict exclusion of target G3 and period grades G1, G2 in early scenario."""
    bundle = prepare_dataset_bundle(include_prior_grades=False)

    # 1. Feature sets must NOT contain targets
    assert "G3" not in bundle.feature_names
    assert "G1" not in bundle.feature_names
    assert "G2" not in bundle.feature_names

    # 2. X_train and X_test DataFrames must NOT contain targets
    assert "G3" not in bundle.X_train.columns
    assert "G1" not in bundle.X_train.columns
    assert "G2" not in bundle.X_train.columns
    assert "G3" not in bundle.X_test.columns
    assert "G1" not in bundle.X_test.columns
    assert "G2" not in bundle.X_test.columns

    # 3. Features must match early feature set
    assert set(bundle.feature_names) == set(EARLY_FEATURES)


def test_stratified_split_preserves_class_ratio():
    """Verifies train and test partitions maintain identical risk proportions."""
    bundle = prepare_dataset_bundle(test_size=0.20, random_state=42)

    train_rate = bundle.metadata["risk_rate_train"]
    test_rate = bundle.metadata["risk_rate_test"]
    overall_rate = bundle.metadata["risk_rate_overall"]

    # Stratification check (tolerance within 1.5%)
    assert abs(train_rate - overall_rate) < 0.015
    assert abs(test_rate - overall_rate) < 0.015
    assert len(bundle.X_train) == 316
    assert len(bundle.X_test) == 79
