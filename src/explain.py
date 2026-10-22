"""Model Explainability and Feature Attribution Module for EduPredict AI.

Integrates SHAP (SHapley Additive exPlanations) and permutation importance
to provide transparent global feature importance rankings and individualized
student-level predictive factor breakdowns without confusing correlation with causation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance

try:
    from preprocessing import get_clean_feature_names
except ImportError:
    get_clean_feature_names = None  # type: ignore

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Contextual descriptions for top predictive features
FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "failures": "Number of past class failures (strongest historical indicator of academic struggle)",
    "schoolsup_yes": "Receiving extra educational support from school (remedial assistance indicator)",
    "Medu": "Mother's education level (socioeconomic and academic home environment factor)",
    "Fedu": "Father's education level (parental academic foundation)",
    "absences": "Number of recorded school absences (class engagement and attendance indicator)",
    "studytime": "Weekly dedicated study time (commitment to coursework)",
    "goout": "Frequency of going out with peers (social balance vs. study focus)",
    "Dalc": "Workday alcohol consumption (behavioral and lifestyle factor)",
    "Walc": "Weekend alcohol consumption (lifestyle factor impacting recovery and focus)",
    "age": "Student's age (repetition of earlier grade levels or delayed schooling)",
    "higher_yes": "Aspiration to attain higher education (strong motivational driver)",
    "romantic_yes": "Involvement in a romantic relationship",
    "famrel": "Quality of family relationships (socio-emotional stability)",
    "freetime": "Available free time after school hours",
    "health": "Current health status (physical wellbeing)",
    "traveltime": "Commute time between home and school",
    "internet_yes": "Home internet access (educational resource availability)",
    "paid_yes": "Enrolled in extra paid classes within the subject",
    "activities_yes": "Participation in extra-curricular activities",
}


def compute_feature_importances(
    model: Any,
    feature_names: List[str],
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Extracts feature importance using tree MDI, linear coefficients, or permutation importance."""
    importances: Optional[np.ndarray] = None

    # Tree-based model (RandomForest, GradientBoosting, XGBoost)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    # Linear model (LinearRegression, LogisticRegression)
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim > 1:
            coef = coef[0]
        importances = np.abs(coef)
    # Permutation importance fallback
    elif X_val is not None and y_val is not None:
        logger.info("Computing permutation importance as fallback...")
        perm = permutation_importance(model, X_val, y_val, n_repeats=5, random_state=42)
        importances = perm.importances_mean

    if importances is None:
        importances = np.ones(len(feature_names)) / len(feature_names)

    # Normalize to percentage
    total = np.sum(importances)
    norm_importances = (importances / total) if total > 0 else importances

    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": norm_importances,
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    return df_imp


def generate_shap_report(
    model: Any,
    X_train_trans: np.ndarray,
    X_test_trans: np.ndarray,
    feature_names: List[str],
    output_plot_path: Path,
) -> Tuple[Optional[Any], pd.DataFrame]:
    """Generates SHAP summary visualizations and returns global importance DataFrame."""
    df_imp = compute_feature_importances(model, feature_names)

    if not SHAP_AVAILABLE:
        logger.warning("SHAP library not available. Generating standard feature importance plot.")
        _plot_standard_feature_importances(df_imp, output_plot_path)
        return None, df_imp

    try:
        logger.info("Computing SHAP values for model explainability...")
        # Use TreeExplainer for tree models, Explainer for others
        model_type = type(model).__name__.lower()
        if any(t in model_type for t in ["randomforest", "gradientboosting", "xgb"]):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_trans)
        else:
            explainer = shap.LinearExplainer(model, X_train_trans)
            shap_values = explainer.shap_values(X_test_trans)

        # Handle binary classification 2D or list of outputs
        if isinstance(shap_values, list) and len(shap_values) == 2:
            # Positive class (At Risk)
            shap_matrix = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_matrix = shap_values[:, :, 1]
        else:
            shap_matrix = np.array(shap_values)

        # Compute mean absolute SHAP value per feature
        mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
        df_shap = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap,
        }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

        # Plot top 15 features
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

        # Panel 1: Bar chart of top 15 features by SHAP impact
        top15 = df_shap.head(15)
        sns.barplot(
            data=top15,
            x="mean_abs_shap",
            y="feature",
            hue="feature",
            palette="Blues_r",
            legend=False,
            ax=ax1,
            edgecolor="black",
        )
        ax1.set_title("Global Feature Impact (Mean |SHAP Value|)", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Mean Absolute SHAP Value (Impact on Model Output)")
        ax1.set_ylabel("Feature Name")

        # Panel 2: Standard MDI / coefficient importance
        top15_mdi = df_imp.head(15)
        sns.barplot(
            data=top15_mdi,
            x="importance",
            y="feature",
            hue="feature",
            palette="Purples_r",
            legend=False,
            ax=ax2,
            edgecolor="black",
        )
        ax2.set_title("Relative Model Feature Importance (Normalized)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Relative Importance Score")
        ax2.set_ylabel("")

        plt.suptitle("Model Explainability & Key Predictive Drivers (Early Prediction Scenario)", fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout()
        output_plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_plot_path, dpi=300)
        plt.close(fig)
        logger.info(f"Saved SHAP explainability plot to: {output_plot_path}")

        return explainer, df_shap

    except Exception as e:
        logger.warning(f"SHAP computation encountered an issue ({e}). Falling back to standard plot.")
        _plot_standard_feature_importances(df_imp, output_plot_path)
        return None, df_imp


def _plot_standard_feature_importances(df_imp: pd.DataFrame, output_path: Path) -> None:
    """Generates bar chart of top 15 feature importances as robust fallback."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    top15 = df_imp.head(15)

    sns.barplot(
        data=top15,
        x="importance",
        y="feature",
        palette="Blues_r",
        ax=ax,
        edgecolor="black",
    )
    ax.set_title("Top 15 Predictive Features (Relative Importance)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Normalized Feature Importance")
    ax.set_ylabel("Attribute")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved standard feature importance plot to: {output_path}")


def _build_encoded_feature_map(preprocessor: Any) -> Dict[str, Dict[str, Any]]:
    """Builds a lookup from encoded feature name to its original column and metadata.

    For one-hot encoded categorical features:
      - 'sex_M' -> { original_column: 'sex', encoded_category: 'M' }
    For scaled numeric features:
      - 'failures' -> { original_column: 'failures', encoded_category: None }
    """
    mapping: Dict[str, Dict[str, Any]] = {}

    cat_transformer = preprocessor.named_transformers_.get("cat")
    num_transformer = preprocessor.named_transformers_.get("num")

    # Categorical — OHE features
    if cat_transformer is not None:
        ohe = cat_transformer.named_steps.get("onehot")
        cat_cols = preprocessor.transformers_[0][2]  # column list passed to 'cat'
        if ohe is not None and hasattr(ohe, "get_feature_names_out"):
            for enc_name in ohe.get_feature_names_out(cat_cols):
                for orig_col in cat_cols:
                    prefix = orig_col + "_"
                    if enc_name.startswith(prefix):
                        category_val = enc_name[len(prefix):]
                        mapping[enc_name] = {
                            "original_column": orig_col,
                            "encoded_category": category_val,
                        }
                        break

    # Numeric — scaled features keep their names
    if num_transformer is not None:
        num_cols = preprocessor.transformers_[1][2]
        for col in num_cols:
            mapping[col] = {
                "original_column": col,
                "encoded_category": None,
            }

    return mapping


def compute_single_student_shap(
    model: Any,
    preprocessor: Any,
    student_record: Dict[str, Any],
    top_k: int = 5,
) -> Dict[str, Any]:
    """Computes instance-level SHAP attributions for an individual student using the
    saved regression model. SHAP values are mathematically verified to satisfy
    base_value + sum(SHAP) = model_prediction within floating-point precision.

    Returned entries include:
      - encoded_feature: the exact model feature name (e.g. 'sex_M', 'failures')
      - original_column: the raw input column (e.g. 'sex', 'failures')
      - original_input_value: the student's actual submitted value for that column
      - encoded_category: the category this OHE indicator represents (None for numeric)
      - indicator_active: True if the student's value matches this OHE indicator
      - shap_impact: signed SHAP contribution; positive = pushes prediction up

    SHAP contributions reflect what the trained model learned from historical
    training data. They are statistical associations, not causal mechanisms.
    Inactive OHE indicators (is_active=False) can still carry non-zero SHAP
    values because Random Forest SHAP is computed over the full decision path,
    not only the active leaf.
    """
    if not SHAP_AVAILABLE:
        return {
            "methodology": "SHAP Unavailable",
            "status": "SHAP library not installed; run: pip install shap",
            "model_base_value": None,
            "reconstructed_prediction": None,
            "shap_additivity_verified": False,
            "factors_pushing_grade_up": [],
            "factors_pushing_grade_down": [],
        }

    try:
        # Build feature metadata map once
        enc_map = _build_encoded_feature_map(preprocessor)
        feature_cols = preprocessor.feature_names_in_.tolist()

        # Strict column order as the preprocessor expects
        row_dict = {col: student_record[col] for col in feature_cols if col in student_record}
        missing = [col for col in feature_cols if col not in student_record]
        if missing:
            raise ValueError(f"Student record missing columns required by preprocessor: {missing}")

        row_df = pd.DataFrame([row_dict])
        X_trans = preprocessor.transform(row_df)

        # Retrieve clean feature names aligned with X_trans columns
        clean_names = get_clean_feature_names(preprocessor)

        model_type = type(model).__name__.lower()
        if any(t in model_type for t in ["randomforest", "gradientboosting", "xgb", "xgbreg"]):
            explainer = shap.TreeExplainer(model)
            shap_raw = explainer.shap_values(X_trans)
        else:
            # LinearExplainer requires a representative background dataset;
            # we use the single observation as background (best available without
            # storing training data; note: this is an approximation for linear models)
            explainer = shap.Explainer(model, X_trans)
            shap_raw = explainer(X_trans).values

        # Normalise output to 1-D array
        if isinstance(shap_raw, list):
            shap_vals = np.array(shap_raw[0]).flatten()
        elif isinstance(shap_raw, np.ndarray) and shap_raw.ndim > 1:
            shap_vals = shap_raw[0].flatten()
        else:
            shap_vals = np.array(shap_raw).flatten()

        raw_base = explainer.expected_value
        if isinstance(raw_base, (np.ndarray, list)):
            base_val_scalar = float(np.array(raw_base).flat[0])
        else:
            base_val_scalar = float(raw_base)

        model_pred = float(model.predict(X_trans)[0])
        reconstructed = base_val_scalar + float(np.sum(shap_vals))
        additivity_ok = abs(reconstructed - model_pred) < 1e-3

        # Build per-feature records with correct metadata
        paired = []
        for i, enc_name in enumerate(clean_names):
            info = enc_map.get(enc_name, {"original_column": enc_name, "encoded_category": None})
            orig_col = info["original_column"]
            enc_cat = info["encoded_category"]
            raw_val = student_record.get(orig_col, "N/A")

            # Determine if this OHE indicator is the active one for this student
            if enc_cat is not None:
                indicator_active = (str(raw_val) == enc_cat)
            else:
                indicator_active = True  # numeric features always apply

            paired.append({
                "encoded_feature": enc_name,
                "original_column": orig_col,
                "encoded_category": enc_cat,
                "original_input_value": raw_val,
                "indicator_active": indicator_active,
                "shap_impact": round(float(shap_vals[i]), 4),
            })

        # Sort into positive and negative contributions
        positives = sorted(
            [p for p in paired if p["shap_impact"] > 0],
            key=lambda x: x["shap_impact"],
            reverse=True,
        )[:top_k]

        negatives = sorted(
            [p for p in paired if p["shap_impact"] < 0],
            key=lambda x: x["shap_impact"],
        )[:top_k]

        return {
            "methodology": "SHAP (SHapley Additive exPlanations) — TreeExplainer on Regression Model",
            "model_base_value": round(base_val_scalar, 4),
            "model_prediction": round(model_pred, 4),
            "reconstructed_prediction": round(reconstructed, 4),
            "shap_additivity_verified": additivity_ok,
            "factors_pushing_grade_up": positives,
            "factors_pushing_grade_down": negatives,
            "interpretation_note": (
                "SHAP values measure each feature's marginal contribution to the prediction "
                "relative to the training-set average. A positive SHAP pushes the prediction "
                "above the average grade; a negative SHAP pushes it below. "
                "Inactive OHE indicators (indicator_active=False) can still carry non-zero SHAP "
                "because tree-based SHAP propagates contributions across the full decision path. "
                "SHAP reflects the model's learned associations in historical data, not causation."
            ),
            "disclaimer": (
                "These attributions describe what the trained model uses to form its prediction. "
                "They must not be interpreted as proof of causal relationships or used to label students."
            ),
        }

    except Exception as e:
        logger.warning(f"Instance SHAP computation failed: {e}")
        return {
            "methodology": "SHAP Computation Failed",
            "error": str(e),
            "model_base_value": None,
            "reconstructed_prediction": None,
            "shap_additivity_verified": False,
            "factors_pushing_grade_up": [],
            "factors_pushing_grade_down": [],
        }


def explain_student_prediction(
    student_record: Dict[str, Any],
    predicted_grade: float,
    risk_prob: float,
    model: Optional[Any] = None,
    preprocessor: Optional[Any] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Generates an integrated explanation combining SHAP attributions with
    rule-based academic counselling guidance.

    Clearly separates two distinct explanation sources:
      1. model_shap_attributions  — computed from the actual model prediction via
         TreeExplainer (SHAP). Reflects what the model uses, not causation.
      2. rule_based_guidance      — heuristic rules applied to academic indicators
         (prior failures, absences, study time, post-secondary plans). Explicitly
         labelled as institutional guidance, not ML output.

    Rules are derived only from objective academic indicators. Sensitive
    demographic characteristics such as sex, family structure, parental
    occupation, alcohol habits, or guardian type are NOT used as labelling
    criteria and are NOT shown in rule-based flags. Those attributes may still
    appear in SHAP output because the model internally learned statistical
    associations from historical data — that is labelled explicitly and
    accompanied by a causal disclaimer.
    """
    # ── 1. SHAP attributions (model-derived, mathematically grounded) ──
    shap_explanation: Optional[Dict[str, Any]] = None
    if model is not None and preprocessor is not None:
        shap_explanation = compute_single_student_shap(
            model=model,
            preprocessor=preprocessor,
            student_record=student_record,
            top_k=top_k,
        )

    # ── 2. Rule-based academic guidance (heuristic, non-model) ──────────
    # Only academic indicators are used. No demographic or lifestyle labelling.
    rule_risk_flags: List[Dict[str, str]] = []
    rule_protective_flags: List[Dict[str, str]] = []

    # Prior course failures
    failures = int(student_record.get("failures", 0))
    if failures > 0:
        rule_risk_flags.append({
            "factor": f"Prior course difficulties ({failures} previous failure(s) recorded)",
            "category": "Academic Background",
            "note": "Rule-based flag (not a SHAP value). Recommends foundational concept review.",
        })
    else:
        rule_protective_flags.append({
            "factor": "No prior course failures recorded",
            "category": "Academic Background",
            "note": "Rule-based flag. Standard academic progression with no recorded prior difficulties.",
        })

    # Class attendance
    absences = int(student_record.get("absences", 0))
    if absences >= 10:
        rule_risk_flags.append({
            "factor": f"High recorded absences ({absences} school days)",
            "category": "Attendance",
            "note": "Rule-based flag. High absenteeism may create conceptual gaps; attendance support recommended.",
        })
    elif absences <= 2:
        rule_protective_flags.append({
            "factor": f"Strong attendance record ({absences} absences)",
            "category": "Attendance",
            "note": "Rule-based flag. Consistent class attendance supports continuous learning.",
        })

    # Weekly self-study time
    studytime = int(student_record.get("studytime", 2))
    studytime_labels = {1: "< 2 hours/week", 2: "2–5 hours/week", 3: "5–10 hours/week", 4: "> 10 hours/week"}
    if studytime == 1:
        rule_risk_flags.append({
            "factor": f"Weekly self-study: {studytime_labels.get(studytime, str(studytime))}",
            "category": "Study Habits",
            "note": "Rule-based flag. Low study time; structured study schedule support recommended.",
        })
    elif studytime >= 3:
        rule_protective_flags.append({
            "factor": f"Weekly self-study: {studytime_labels.get(studytime, str(studytime))}",
            "category": "Study Habits",
            "note": "Rule-based flag. Dedicated independent study habit noted.",
        })

    # Post-secondary educational plans
    higher = str(student_record.get("higher", "yes")).lower()
    if higher == "yes":
        rule_protective_flags.append({
            "factor": "Student reports plans for post-secondary education",
            "category": "Academic Motivation",
            "note": "Rule-based flag. Long-term educational aspiration noted.",
        })
    else:
        rule_risk_flags.append({
            "factor": "No current post-secondary education plan reported",
            "category": "Academic Motivation",
            "note": "Rule-based flag. Academic and career mentoring may help connect coursework to vocational goals.",
        })

    # ── 3. Actionable counselling recommendations (rule-based) ──────────
    schoolsup = str(student_record.get("schoolsup", "no")).lower()
    actionable_recs = generate_counseling_recommendations(
        failures=failures,
        absences=absences,
        studytime=studytime,
        schoolsup=schoolsup,
        predicted_grade=predicted_grade,
        risk_prob=risk_prob,
    )

    result_payload: Dict[str, Any] = {
        "predicted_grade": predicted_grade,
        "risk_probability": round(risk_prob, 4),
        "supportive_intervention_recommended": risk_prob >= 0.50 or predicted_grade < 10.0,
        "rule_based_risk_flags": rule_risk_flags[:top_k],
        "rule_based_protective_flags": rule_protective_flags[:top_k],
        "supportive_counseling_guidance": {
            "source": "Rule-based institutional guidance heuristics (not ML output)",
            "recommendations": actionable_recs,
        },
        "ethical_disclaimer": (
            "These explanations and recommendations are advisory tools to support "
            "educational mentoring. They must not be used for disciplinary action, "
            "grade decisions, or deterministic labelling of students."
        ),
    }

    if shap_explanation is not None:
        result_payload["model_shap_attributions"] = shap_explanation

    # ── Backward-compatibility aliases for existing tests ────────────────
    result_payload["primary_risk_factors"] = rule_risk_flags[:top_k]
    result_payload["protective_factors"] = rule_protective_flags[:top_k]
    result_payload["actionable_recommendations"] = actionable_recs
    result_payload["disclaimer"] = result_payload["ethical_disclaimer"]

    return result_payload


def generate_counseling_recommendations(
    failures: int,
    absences: int,
    studytime: int,
    schoolsup: str,
    predicted_grade: float,
    risk_prob: float,
) -> List[str]:
    """Generates constructive, non-stigmatizing intervention actions for academic advisors."""
    recommendations: List[str] = []

    if risk_prob >= 0.50 or predicted_grade < 10.0:
        recommendations.append("Proactive Mentoring: Schedule an informal check-in with the academic advisor to co-design a personalized study plan.")

    if failures > 0:
        recommendations.append("Foundational Support: Connect student with peer tutoring or math clinic sessions to reinforce core concepts.")

    if absences >= 8:
        recommendations.append("Attendance Encouragement: Coordinate with student counselor to explore any commuting, health, or scheduling hurdles.")

    if studytime <= 2:
        recommendations.append("Time-Management Coaching: Provide weekly planner templates and encourage structured study blocks in the library.")

    if schoolsup == "no" and predicted_grade < 11.0:
        recommendations.append("Resource Access: Encourage student to explore school-provided review workshops and office hours.")

    if not recommendations:
        recommendations.append("Academic Affirmation: Encourage student to maintain current productive study habits and extracurricular balance.")

    return recommendations
