"""Tests for saved model artifact loading, prediction consistency, probability output,
and SHAP-based explanation consistency.

These tests verify the full saved-to-loaded lifecycle to ensure model artifacts
are recoverable in a fresh Python process and produce consistent results.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

MODELS_DIR = Path(__file__).parent.parent / "models"


# ─────────────────────────── fixtures ──────────────────────────

@pytest.fixture(scope="module")
def artifact_paths():
    """Collects and validates artifact paths before tests run."""
    return {
        "metadata":       MODELS_DIR / "model_metadata.json",
        "preprocessor":   MODELS_DIR / "preprocessor.joblib",
        "regressor":      MODELS_DIR / "student_grade_regressor.joblib",
        "classifier":     MODELS_DIR / "student_risk_classifier.joblib",
        "full_reg":       MODELS_DIR / "full_regression_pipeline.joblib",
        "full_clf":       MODELS_DIR / "full_classification_pipeline.joblib",
    }


@pytest.fixture(scope="module")
def loaded_artifacts(artifact_paths):
    """Loads every artifact exactly once per test session."""
    for name, path in artifact_paths.items():
        if not path.exists():
            pytest.skip(f"Artifact '{name}' missing at {path} — run train_models.py first.")

    with open(artifact_paths["metadata"], encoding="utf-8") as f:
        metadata = json.load(f)

    return {
        "metadata":     metadata,
        "preprocessor": joblib.load(artifact_paths["preprocessor"]),
        "regressor":    joblib.load(artifact_paths["regressor"]),
        "classifier":   joblib.load(artifact_paths["classifier"]),
        "full_reg":     joblib.load(artifact_paths["full_reg"]),
        "full_clf":     joblib.load(artifact_paths["full_clf"]),
    }


@pytest.fixture(scope="module")
def sample_records():
    from predict import get_sample_student
    return {
        "safe":    get_sample_student(at_risk=False),
        "at_risk": get_sample_student(at_risk=True),
    }


# ─────────────────────── Artifact Loading Tests ─────────────────────────

class TestArtifactLoading:
    """Verifies that every saved artifact loads without error and prints its path."""

    def test_all_artifacts_exist(self, artifact_paths):
        missing = [name for name, p in artifact_paths.items() if not p.exists()]
        assert missing == [], f"Missing artifacts: {missing}"

    def test_metadata_json_structure(self, loaded_artifacts):
        meta = loaded_artifacts["metadata"]
        assert "selected_models" in meta
        assert "regression" in meta["selected_models"]
        assert "classification" in meta["selected_models"]
        assert "features" in meta
        assert "raw_features" in meta["features"]
        assert len(meta["features"]["raw_features"]) == 30

    def test_preprocessor_has_expected_output_shape(self, loaded_artifacts, sample_records):
        prep = loaded_artifacts["preprocessor"]
        from predict import get_sample_student
        student = get_sample_student()
        required = loaded_artifacts["metadata"]["features"]["raw_features"]
        row_df = pd.DataFrame([{c: student[c] for c in required}])
        X = prep.transform(row_df)
        assert X.shape == (1, 56), f"Unexpected transformed shape: {X.shape}"

    def test_full_pipelines_load_and_predict(self, loaded_artifacts, sample_records):
        required = loaded_artifacts["metadata"]["features"]["raw_features"]
        row_df = pd.DataFrame([{c: sample_records["at_risk"][c] for c in required}])

        reg_pred = loaded_artifacts["full_reg"].predict(row_df)
        clf_pred = loaded_artifacts["full_clf"].predict(row_df)

        assert len(reg_pred) == 1
        assert 0.0 <= float(reg_pred[0]) <= 20.0
        assert int(clf_pred[0]) in [0, 1]

    def test_artifact_paths_logged_by_predictor(self):
        """Verifies EduPredictor reports exact artifact paths on load."""
        from predict import EduPredictor
        predictor = EduPredictor(models_dir=MODELS_DIR)
        paths = predictor.get_artifact_paths()
        for key in ["metadata", "preprocessor", "regressor", "classifier"]:
            assert key in paths
            assert Path(paths[key]).exists(), f"Artifact path '{key}' does not exist: {paths[key]}"


# ─────────────────────── Prediction Consistency Tests ────────────────────

class TestPredictionConsistency:
    """Verifies that predictions are reproducible and numerically stable."""

    def test_same_input_gives_same_prediction(self, loaded_artifacts, sample_records):
        """Same student record submitted twice must produce bit-exact same prediction."""
        prep = loaded_artifacts["preprocessor"]
        reg  = loaded_artifacts["regressor"]
        clf  = loaded_artifacts["classifier"]

        required = loaded_artifacts["metadata"]["features"]["raw_features"]
        row_df = pd.DataFrame([{c: sample_records["safe"][c] for c in required}])

        X1 = prep.transform(row_df)
        X2 = prep.transform(row_df)

        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(reg.predict(X1), reg.predict(X2))
        np.testing.assert_array_equal(clf.predict(X1), clf.predict(X2))

    def test_at_risk_student_predicts_lower_grade(self, loaded_artifacts, sample_records):
        """At-risk student should receive a predicted grade lower than safe student."""
        from predict import EduPredictor
        predictor = EduPredictor(models_dir=MODELS_DIR)
        res_safe    = predictor.predict_single(sample_records["safe"],    include_explanation=False)
        res_at_risk = predictor.predict_single(sample_records["at_risk"], include_explanation=False)

        assert res_at_risk["predicted_grade"] < res_safe["predicted_grade"], (
            f"At-risk student grade ({res_at_risk['predicted_grade']}) should be less than "
            f"safe student grade ({res_safe['predicted_grade']})"
        )

    def test_grade_bounds_always_respected(self, loaded_artifacts, sample_records):
        """Every predicted grade must be within the valid [0.0, 20.0] range."""
        from predict import EduPredictor
        predictor = EduPredictor(models_dir=MODELS_DIR)
        for label, student in sample_records.items():
            res = predictor.predict_single(student, include_explanation=False)
            assert 0.0 <= res["predicted_grade"] <= 20.0, (
                f"Grade {res['predicted_grade']} for '{label}' is outside [0, 20]"
            )

    def test_configurable_risk_threshold(self, sample_records):
        """Lowering risk_threshold to 0.0 should classify every student as At Risk."""
        from predict import EduPredictor
        predictor = EduPredictor(models_dir=MODELS_DIR, risk_threshold=0.0)
        for label, student in sample_records.items():
            res = predictor.predict_single(student, include_explanation=False, risk_threshold=0.0)
            assert res["academic_risk_class"] == 1, (
                f"With threshold 0.0, '{label}' should be classified as At Risk"
            )


# ─────────────────────── Probability Output Tests ────────────────────────

class TestProbabilityOutput:
    """Verifies risk probability outputs are valid and reasonably calibrated."""

    def test_probabilities_are_within_unit_interval(self, loaded_artifacts, sample_records):
        from predict import EduPredictor
        predictor = EduPredictor(models_dir=MODELS_DIR)
        for label, student in sample_records.items():
            res = predictor.predict_single(student, include_explanation=False)
            prob = res["risk_probability"]
            assert 0.0 <= prob <= 1.0, f"Risk probability {prob} for '{label}' is out of [0, 1]"

    def test_at_risk_student_has_higher_probability(self, loaded_artifacts, sample_records):
        from predict import EduPredictor
        predictor = EduPredictor(models_dir=MODELS_DIR)
        safe_prob    = predictor.predict_single(sample_records["safe"],    include_explanation=False)["risk_probability"]
        at_risk_prob = predictor.predict_single(sample_records["at_risk"], include_explanation=False)["risk_probability"]

        assert at_risk_prob > safe_prob, (
            f"At-risk student probability ({at_risk_prob}) should exceed "
            f"safe student probability ({safe_prob})"
        )

    def test_brier_score_present_in_saved_metrics(self):
        """Verifies Brier score is stored for the selected classifier."""
        metrics_path = Path(__file__).parent.parent / "outputs" / "metrics" / "classification_metrics.json"
        if not metrics_path.exists():
            pytest.skip("Classification metrics file not found.")
        with open(metrics_path, encoding="utf-8") as f:
            clf_data = json.load(f)
        test_evals = clf_data.get("test_evaluation", {})
        for model_name, metrics in test_evals.items():
            brier = metrics.get("Brier_Score")
            if brier is not None:
                assert 0.0 <= brier <= 1.0, f"Brier score {brier} for '{model_name}' is invalid"


# ─────────────────────── Explanation Consistency Tests ───────────────────

class TestExplanationConsistency:
    """Verifies that SHAP attributions are internally consistent with model predictions."""

    def test_shap_explanation_present_in_prediction(self, sample_records):
        from predict import EduPredictor
        predictor = EduPredictor(models_dir=MODELS_DIR)
        res = predictor.predict_single(sample_records["at_risk"], include_explanation=True)

        assert "explanation" in res
        exp = res["explanation"]

        # Required payload keys
        for key in ["actionable_recommendations", "primary_risk_factors",
                    "protective_factors", "ethical_disclaimer"]:
            assert key in exp, f"Missing explanation key: '{key}'"

        # Recommendations must be non-empty
        assert len(exp["actionable_recommendations"]) > 0

    def test_shap_attributions_reconstruct_prediction(self, sample_records, loaded_artifacts):
        """Verifies base_value + sum(SHAP) ≈ model prediction for regression."""
        try:
            import shap
        except ImportError:
            pytest.skip("SHAP library not available.")

        from preprocessing import get_clean_feature_names, transform_data
        prep = loaded_artifacts["preprocessor"]
        reg  = loaded_artifacts["regressor"]
        required = loaded_artifacts["metadata"]["features"]["raw_features"]

        student = sample_records["at_risk"]
        row_df = pd.DataFrame([{c: student[c] for c in required}])
        X_trans, _ = transform_data(prep, row_df)

        explainer = shap.TreeExplainer(reg)
        shap_values = explainer.shap_values(X_trans)
        base_val = float(explainer.expected_value) if not hasattr(explainer.expected_value, "__len__") \
            else float(explainer.expected_value[0])

        shap_arr = np.array(shap_values).flatten()
        reconstructed = base_val + float(np.sum(shap_arr))
        model_pred = float(reg.predict(X_trans)[0])

        assert abs(reconstructed - model_pred) < 1e-3, (
            f"SHAP decomposition mismatch: reconstructed={reconstructed:.4f}, "
            f"model_pred={model_pred:.4f}, diff={abs(reconstructed - model_pred):.6f}"
        )

    def test_shap_method_labeled_correctly(self, sample_records):
        """Verifies SHAP explanations are labeled as SHAP, not rule-based."""
        from predict import EduPredictor
        predictor = EduPredictor(models_dir=MODELS_DIR)
        res = predictor.predict_single(sample_records["at_risk"], include_explanation=True)
        exp = res["explanation"]

        if "model_shap_attributions" in exp:
            methodology = exp["model_shap_attributions"].get("methodology", "")
            assert "SHAP" in methodology, (
                f"SHAP block should declare SHAP methodology, got: '{methodology}'"
            )

    def test_shap_feature_mapping_is_correct(self, sample_records):
        """Verifies that each SHAP factor carries original_column and original_input_value."""
        from predict import EduPredictor
        predictor = EduPredictor(models_dir=MODELS_DIR)
        res = predictor.predict_single(sample_records["at_risk"], include_explanation=True)
        exp = res["explanation"]

        if "model_shap_attributions" not in exp:
            pytest.skip("SHAP not available in this environment.")

        shap_block = exp["model_shap_attributions"]
        # Check additivity was verified
        assert shap_block.get("shap_additivity_verified", False), (
            "SHAP additivity check failed: base_value + sum(shap) != model prediction"
        )

        for direction in ["factors_pushing_grade_up", "factors_pushing_grade_down"]:
            for factor in shap_block.get(direction, []):
                assert "encoded_feature" in factor, f"Missing 'encoded_feature' in {direction}"
                assert "original_column" in factor, f"Missing 'original_column' in {direction}"
                assert "original_input_value" in factor, f"Missing 'original_input_value' in {direction}"
                assert "indicator_active" in factor, f"Missing 'indicator_active' in {direction}"
                assert "shap_impact" in factor, f"Missing 'shap_impact' in {direction}"
                # Verify sign is correct
                if direction == "factors_pushing_grade_up":
                    assert factor["shap_impact"] > 0, f"Positive factor has non-positive SHAP: {factor}"
                else:
                    assert factor["shap_impact"] < 0, f"Negative factor has non-negative SHAP: {factor}"

    def test_explanation_disclaimer_is_present_and_ethical(self, sample_records):
        """Checks that the ethical disclaimer is always present and non-empty."""
        from predict import EduPredictor
        predictor = EduPredictor(models_dir=MODELS_DIR)
        for label, student in sample_records.items():
            res = predictor.predict_single(student, include_explanation=True)
            disclaimer = res["explanation"].get("ethical_disclaimer", "") or \
                         res["explanation"].get("disclaimer", "")
            assert len(disclaimer) > 20, (
                f"Ethical disclaimer for '{label}' is missing or too short."
            )
