"""Reusable production-grade inference interface for EduPredict AI.

Loads serialized artifacts, validates student feature schemas,
prevents data leakage, generates grade predictions and risk probabilities,
and provides individualized explainability drivers for counseling interventions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from data_loader import (
    EARLY_FEATURES,
    NUMERIC_FEATURES_EARLY,
    NUMERIC_RANGES,
    VALID_CATEGORICAL_VALUES,
)
from explain import explain_student_prediction

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class PredictionValidationError(ValueError):
    """Raised when inference input data violates required schema or integrity constraints."""
    pass


class EduPredictor:
    """Production inference engine for student grade and academic risk predictions."""

    def __init__(
        self,
        models_dir: Optional[str | Path] = None,
        risk_threshold: float = 0.50,
    ) -> None:
        """Initializes predictor by loading persisted model artifacts and metadata."""
        self.risk_threshold = float(risk_threshold)
        self.models_dir = self._resolve_models_dir(models_dir)
        self.artifact_paths: Dict[str, Path] = {}
        self._load_artifacts()

    @staticmethod
    def _resolve_models_dir(custom_path: Optional[str | Path] = None) -> Path:
        """Resolves models directory from given path or default project roots."""
        if custom_path is not None:
            p = Path(custom_path)
            if p.exists():
                return p.resolve()
            raise FileNotFoundError(f"Specified models directory does not exist: {custom_path}")

        candidates = [
            Path("models"),
            Path("AIML PROJECT 2/models"),
            Path("d:/7th Sem/Artifical Intelligence and Machine Learning/AIML PROJECT 2/models"),
        ]
        for cand in candidates:
            if (cand / "model_metadata.json").exists():
                return cand.resolve()

        raise FileNotFoundError(f"Could not locate trained models directory. Checked: {[str(c) for c in candidates]}")

    def _load_artifacts(self) -> None:
        """Loads preprocessor, regressor, classifier, and metadata into memory."""
        meta_file = self.models_dir / "model_metadata.json"
        prep_file = self.models_dir / "preprocessor.joblib"
        reg_file = self.models_dir / "student_grade_regressor.joblib"
        clf_file = self.models_dir / "student_risk_classifier.joblib"

        self.artifact_paths = {
            "metadata": meta_file.resolve(),
            "preprocessor": prep_file.resolve(),
            "regressor": reg_file.resolve(),
            "classifier": clf_file.resolve(),
        }

        # Validate existence of every artifact with clear error reporting
        for name, required_file in self.artifact_paths.items():
            if not required_file.exists():
                raise FileNotFoundError(
                    f"Missing required model artifact '{name}' at: {required_file}.\n"
                    f"Please run 'python src/train_models.py' to train and persist all required model files."
                )

        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                self.metadata: Dict[str, Any] = json.load(f)

            self.preprocessor = joblib.load(prep_file)
            self.regressor = joblib.load(reg_file)
            self.classifier = joblib.load(clf_file)
        except Exception as err:
            raise RuntimeError(f"Failed to load serialized model artifacts from {self.models_dir}: {err}") from err

        self.required_features: List[str] = self.metadata["features"]["raw_features"]
        self.scenario: str = self.metadata.get("scenario", "early_prediction")

        logger.info(f"EduPredictor initialized successfully from {self.models_dir}\n"
                    f"  Artifacts Loaded:\n"
                    f"    - Preprocessor: {self.artifact_paths['preprocessor']}\n"
                    f"    - Regressor:    {self.artifact_paths['regressor']}\n"
                    f"    - Classifier:   {self.artifact_paths['classifier']}\n"
                    f"    - Metadata:     {self.artifact_paths['metadata']}\n"
                    f"  Configured Threshold: {self.risk_threshold:.2f} (Scenario: {self.scenario})")

    def get_artifact_paths(self) -> Dict[str, str]:
        """Returns string representation of all loaded model artifact paths."""
        return {k: str(v) for k, v in self.artifact_paths.items()}

    def validate_student_record(self, record: Dict[str, Any]) -> None:
        """Validates that a single input student dictionary adheres to the schema."""
        if not isinstance(record, dict):
            raise PredictionValidationError(f"Expected dict record, received: {type(record).__name__}")

        # Check for missing required features
        missing_keys = [feat for feat in self.required_features if feat not in record]
        if missing_keys:
            raise PredictionValidationError(
                f"Input student record is missing required features: {missing_keys}"
            )

        # Validate categorical attributes if present in dictionary
        for cat_col, valid_opts in VALID_CATEGORICAL_VALUES.items():
            if cat_col in record:
                val = str(record[cat_col])
                if val not in valid_opts:
                    raise PredictionValidationError(
                        f"Invalid value for '{cat_col}': '{val}'. Allowed options: {valid_opts}"
                    )

        # Validate numeric bounds
        for num_col, (min_bound, max_bound) in NUMERIC_RANGES.items():
            if num_col in record:
                try:
                    num_val = float(record[num_col])
                except (ValueError, TypeError):
                    raise PredictionValidationError(
                        f"Non-numeric value provided for '{num_col}': {record[num_col]}"
                    )
                if num_val < min_bound or num_val > max_bound:
                    raise PredictionValidationError(
                        f"Value for '{num_col}' ({num_val}) is outside valid range [{min_bound}, {max_bound}]."
                    )

    def predict_single(
        self,
        student_record: Dict[str, Any],
        include_explanation: bool = True,
        risk_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Predicts academic final grade, risk status, and intervention guidance for one student."""
        # Validate schema
        self.validate_student_record(student_record)

        effective_threshold = float(risk_threshold if risk_threshold is not None else self.risk_threshold)

        # Prepare 1-row DataFrame with strict column order
        row_df = pd.DataFrame([{col: student_record[col] for col in self.required_features}])

        # Transform using fitted preprocessor
        X_trans = self.preprocessor.transform(row_df)

        # Regression prediction (Grade 0 to 20 scale)
        raw_pred_grade = float(self.regressor.predict(X_trans)[0])
        bounded_grade = round(float(np.clip(raw_pred_grade, 0.0, 20.0)), 2)

        # Classification prediction (Academic Risk)
        if hasattr(self.classifier, "predict_proba"):
            risk_prob = round(float(self.classifier.predict_proba(X_trans)[0, 1]), 4)
            risk_class = int(1 if risk_prob >= effective_threshold else 0)
        else:
            risk_class = int(self.classifier.predict(X_trans)[0])
            risk_prob = 1.0 if risk_class == 1 else 0.0

        # Risk tiering
        if risk_prob >= effective_threshold or bounded_grade < 10.0:
            risk_tier = "High Risk (Fail Expected)"
            risk_status = "At Risk"
        elif risk_prob >= 0.35 or bounded_grade < 11.5:
            risk_tier = "Moderate Caution"
            risk_status = "Borderline / Needs Monitoring"
        else:
            risk_tier = "Low Risk / On Track"
            risk_status = "Safe"

        result: Dict[str, Any] = {
            "predicted_grade": bounded_grade,
            "grade_scale": "0 - 20 (Passing threshold: 10.0)",
            "academic_risk_class": risk_class,
            "risk_status": risk_status,
            "risk_probability": risk_prob,
            "risk_tier": risk_tier,
            "risk_decision_threshold": effective_threshold,
            "prediction_scenario": self.scenario,
        }

        if include_explanation:
            explanation = explain_student_prediction(
                student_record=student_record,
                predicted_grade=bounded_grade,
                risk_prob=risk_prob,
                model=self.regressor,
                preprocessor=self.preprocessor,
            )
            result["explanation"] = explanation

        return result

    def predict_batch(
        self,
        students_df: pd.DataFrame,
        risk_threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        """Generates predictions for a DataFrame of students, returning an enriched DataFrame."""
        if not isinstance(students_df, pd.DataFrame):
            raise PredictionValidationError(f"Expected pandas DataFrame, received: {type(students_df).__name__}")

        missing_cols = [c for c in self.required_features if c not in students_df.columns]
        if missing_cols:
            raise PredictionValidationError(f"Batch DataFrame missing required columns: {missing_cols}")

        effective_threshold = float(risk_threshold if risk_threshold is not None else self.risk_threshold)

        df_subset = students_df[self.required_features].copy()
        X_trans = self.preprocessor.transform(df_subset)

        # Predictions
        pred_grades = np.clip(self.regressor.predict(X_trans), 0.0, 20.0).round(2)

        if hasattr(self.classifier, "predict_proba"):
            risk_probs = self.classifier.predict_proba(X_trans)[:, 1].round(4)
            pred_risk_classes = (risk_probs >= effective_threshold).astype(int)
        else:
            pred_risk_classes = self.classifier.predict(X_trans).astype(int)
            risk_probs = pred_risk_classes.astype(float)

        # Create output DataFrame
        out_df = students_df.copy()
        out_df["predicted_grade"] = pred_grades
        out_df["academic_risk_class"] = pred_risk_classes
        out_df["risk_probability"] = risk_probs
        out_df["risk_status"] = np.where(risk_probs >= effective_threshold, "At Risk", "Safe")
        out_df["risk_tier"] = np.where(
            (risk_probs >= effective_threshold) | (pred_grades < 10.0),
            "High Risk",
            np.where(risk_probs >= 0.35, "Moderate Caution", "Low Risk")
        )

        return out_df


def get_sample_student(at_risk: bool = False) -> Dict[str, Any]:
    """Returns a realistic sample student record for quick testing and demonstrations."""
    if at_risk:
        return {
            "school": "GP",
            "sex": "M",
            "age": 18,
            "address": "R",
            "famsize": "GT3",
            "Pstatus": "A",
            "Medu": 1,
            "Fedu": 1,
            "Mjob": "other",
            "Fjob": "other",
            "reason": "course",
            "guardian": "other",
            "traveltime": 3,
            "studytime": 1,
            "failures": 2,
            "schoolsup": "yes",
            "famsup": "no",
            "paid": "no",
            "activities": "no",
            "nursery": "no",
            "higher": "no",
            "internet": "no",
            "romantic": "yes",
            "famrel": 2,
            "freetime": 4,
            "goout": 4,
            "Dalc": 3,
            "Walc": 4,
            "health": 2,
            "absences": 16,
        }
    else:
        return {
            "school": "GP",
            "sex": "F",
            "age": 16,
            "address": "U",
            "famsize": "LE3",
            "Pstatus": "T",
            "Medu": 4,
            "Fedu": 4,
            "Mjob": "teacher",
            "Fjob": "health",
            "reason": "reputation",
            "guardian": "mother",
            "traveltime": 1,
            "studytime": 3,
            "failures": 0,
            "schoolsup": "no",
            "famsup": "yes",
            "paid": "yes",
            "activities": "yes",
            "nursery": "yes",
            "higher": "yes",
            "internet": "yes",
            "romantic": "no",
            "famrel": 5,
            "freetime": 3,
            "goout": 2,
            "Dalc": 1,
            "Walc": 1,
            "health": 5,
            "absences": 2,
        }


if __name__ == "__main__":
    predictor = EduPredictor()

    print("\n--- TEST 1: High Performing Student Prediction ---")
    good_student = get_sample_student(at_risk=False)
    good_res = predictor.predict_single(good_student)
    print(json.dumps(good_res, indent=2))

    print("\n--- TEST 2: At-Risk Student Prediction ---")
    risk_student = get_sample_student(at_risk=True)
    risk_res = predictor.predict_single(risk_student)
    print(json.dumps(risk_res, indent=2))
