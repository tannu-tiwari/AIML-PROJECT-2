"""Unit tests for model explainability and educational guidance generation."""

import sys
from pathlib import Path
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from explain import (
    FEATURE_DESCRIPTIONS,
    compute_feature_importances,
    explain_student_prediction,
    generate_counseling_recommendations,
)
from predict import get_sample_student


def test_feature_descriptions_coverage():
    """Verifies descriptive dictionary covers all critical predictors."""
    critical_features = ["failures", "absences", "studytime", "Medu", "Fedu", "higher_yes"]
    for feat in critical_features:
        assert feat in FEATURE_DESCRIPTIONS
        assert len(FEATURE_DESCRIPTIONS[feat]) > 10


def test_explain_student_prediction_structure():
    """Tests student explanation payload contains required risk and protective drivers."""
    student = get_sample_student(at_risk=True)
    explanation = explain_student_prediction(
        student_record=student,
        predicted_grade=8.5,
        risk_prob=0.88,
    )

    assert "primary_risk_factors" in explanation
    assert "protective_factors" in explanation
    assert "actionable_recommendations" in explanation
    assert "disclaimer" in explanation

    # At-risk student must have risk factors flagged
    assert len(explanation["primary_risk_factors"]) > 0


def test_generate_counseling_recommendations():
    """Verifies that targeted supportive recommendations trigger for high-risk indicators."""
    recs = generate_counseling_recommendations(
        failures=2,
        absences=15,
        studytime=1,
        schoolsup="no",
        predicted_grade=7.5,
        risk_prob=0.92,
    )

    # Must return at least 4 recommendations for combined high-risk scenario
    assert len(recs) >= 4

    rec_text = " ".join(recs)
    # Verify each major risk dimension triggers a recommendation
    assert any("advisor" in r.lower() or "mentor" in r.lower() for r in recs), \
        "High risk_prob should trigger a mentoring/advisor recommendation"
    assert any("tutor" in r.lower() or "support" in r.lower() or "clinic" in r.lower() for r in recs), \
        "Past failures should trigger a foundational support recommendation"
    assert any("attendance" in r.lower() or "absence" in r.lower() or "counselor" in r.lower() for r in recs), \
        "High absences should trigger an attendance-related recommendation"
    assert any("study" in r.lower() or "time" in r.lower() or "planner" in r.lower() for r in recs), \
        "Low study time should trigger a time-management recommendation"
