"""Unit and integration tests for EduPredictor inference engine."""

import sys
from pathlib import Path
import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from predict import EduPredictor, PredictionValidationError, get_sample_student


@pytest.fixture(scope="module")
def predictor():
    """Initializes EduPredictor instance from serialized models directory."""
    models_dir = Path(__file__).parent.parent / "models"
    return EduPredictor(models_dir=models_dir)


def test_predictor_initialization(predictor):
    """Verifies that predictor loads all production artifacts cleanly."""
    assert predictor.preprocessor is not None
    assert predictor.regressor is not None
    assert predictor.classifier is not None
    assert len(predictor.required_features) == 30
    assert predictor.scenario == "early_prediction"


def test_predict_single_valid_student(predictor):
    """Tests single student prediction for both high performing and at risk students."""
    for at_risk in [False, True]:
        sample = get_sample_student(at_risk=at_risk)
        res = predictor.predict_single(sample, include_explanation=True)

        # Validate response schema
        assert "predicted_grade" in res
        assert "academic_risk_class" in res
        assert "risk_probability" in res
        assert "risk_tier" in res
        assert "explanation" in res

        # Check bounds
        assert 0.0 <= res["predicted_grade"] <= 20.0
        assert res["academic_risk_class"] in [0, 1]
        assert 0.0 <= res["risk_probability"] <= 1.0

        # Validate explanation payload
        exp = res["explanation"]
        assert "primary_risk_factors" in exp
        assert "protective_factors" in exp
        assert "actionable_recommendations" in exp
        assert len(exp["actionable_recommendations"]) > 0


def test_predict_batch_dataframe(predictor):
    """Tests batch prediction using pandas DataFrame."""
    s1 = get_sample_student(at_risk=False)
    s2 = get_sample_student(at_risk=True)
    df_batch = pd.DataFrame([s1, s2])

    res_df = predictor.predict_batch(df_batch)

    assert isinstance(res_df, pd.DataFrame)
    assert len(res_df) == 2
    assert "predicted_grade" in res_df.columns
    assert "academic_risk_class" in res_df.columns
    assert "risk_probability" in res_df.columns
    assert "risk_tier" in res_df.columns

    # High performing student should have higher grade than at risk student
    assert res_df.loc[0, "predicted_grade"] > res_df.loc[1, "predicted_grade"]
    assert res_df.loc[0, "risk_probability"] < res_df.loc[1, "risk_probability"]


def test_missing_column_raises_error(predictor):
    """Confirms that omitting a required feature raises PredictionValidationError."""
    sample = get_sample_student()
    del sample["studytime"]  # Remove a required column

    with pytest.raises(PredictionValidationError) as excinfo:
        predictor.predict_single(sample)

    assert "missing required features" in str(excinfo.value)
    assert "studytime" in str(excinfo.value)


def test_invalid_categorical_value_raises_error(predictor):
    """Confirms invalid categorical value raises PredictionValidationError."""
    sample = get_sample_student()
    sample["school"] = "INVALID_SCHOOL"

    with pytest.raises(PredictionValidationError) as excinfo:
        predictor.predict_single(sample)

    assert "Invalid value for 'school'" in str(excinfo.value)


def test_out_of_bounds_numeric_raises_error(predictor):
    """Confirms out-of-range numeric value raises PredictionValidationError."""
    sample = get_sample_student()
    sample["absences"] = 999  # Out of valid [0, 100] bound

    with pytest.raises(PredictionValidationError) as excinfo:
        predictor.predict_single(sample)

    assert "outside valid range" in str(excinfo.value)


def test_non_numeric_type_raises_error(predictor):
    """Confirms non-numeric string for numeric feature raises PredictionValidationError."""
    sample = get_sample_student()
    sample["failures"] = "none"  # String instead of int

    with pytest.raises(PredictionValidationError) as excinfo:
        predictor.predict_single(sample)

    assert "Non-numeric value provided for 'failures'" in str(excinfo.value)
