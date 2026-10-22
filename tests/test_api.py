"""Focused integration checks for API CSV validation and leakage boundaries."""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api import build_batch_result, build_model_input, dataset_frame, validate_frame  # noqa: E402


def test_bundled_dataset_passes_batch_validation():
    assert validate_frame(dataset_frame().head(3)) == []


def test_invalid_csv_reports_missing_column_and_row_errors():
    frame = dataset_frame().head(1).copy()
    frame["age"] = 99
    frame["Mjob"] = "doctor"
    frame["absences"] = None
    frame = frame.drop(columns=["famsup"])

    issues = validate_frame(frame)
    issue_keys = {(item["row"], item["column"]) for item in issues}
    assert (None, "famsup") in issue_keys
    assert (2, "age") in issue_keys
    assert (2, "Mjob") in issue_keys
    assert (2, "absences") in issue_keys


def test_batch_model_input_excludes_future_grades_and_labels():
    frame = dataset_frame().head(2).copy()
    frame["student_id"] = ["QA-01", "QA-02"]
    model_input = build_model_input(frame)
    assert model_input.columns.tolist() == [
        "school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
        "Mjob", "Fjob", "reason", "guardian", "traveltime", "studytime", "failures",
        "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet",
        "romantic", "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences",
    ]
    assert not {"G1", "G2", "G3", "academic_risk", "student_id"}.intersection(model_input.columns)


def test_batch_result_preserves_identifier_but_labels_observed_grade():
    frame = dataset_frame().head(1).copy()
    frame["student_id"] = "QA-01"
    predictions = pd.DataFrame({
        "predicted_grade": [10.5],
        "academic_risk_class": [0],
        "risk_probability": [0.25],
        "risk_status": ["Safe"],
        "risk_tier": ["Low Risk"],
    })

    result = build_batch_result(frame, predictions)
    assert result.loc[0, "student_id"] == "QA-01"
    assert result.loc[0, "observed_G3"] == frame.loc[0, "G3"]
    assert not {"G1", "G2", "G3"}.intersection(result.columns)

