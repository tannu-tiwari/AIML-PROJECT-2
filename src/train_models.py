"""Model training, cross-validation, selection, and serialization for EduPredict AI.

Executes fair, leakage-free model benchmarking for both regression (grade prediction)
and binary classification (academic risk), selects the best models based on
training-only cross-validation, evaluates once on held-out test data,
and persists production artifacts using Joblib.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from data_loader import DatasetBundle, prepare_dataset_bundle
from evaluate import (
    compute_classification_metrics,
    compute_regression_metrics,
    plot_actual_vs_predicted,
    plot_calibration_curves,
    plot_classification_comparison,
    plot_confusion_matrix_heatmap,
    plot_regression_model_comparison,
    plot_residuals,
    plot_roc_curves,
)
from explain import generate_shap_report
from preprocessing import (
    build_preprocessor,
    fit_preprocessor,
    get_clean_feature_names,
    save_preprocessor,
    transform_data,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_candidate_regressors(random_state: int = 42) -> Dict[str, Any]:
    """Instantiates candidate regression models for academic grade prediction."""
    models: Dict[str, Any] = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10.0),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
            min_samples_split=4,
            random_state=random_state,
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            min_samples_split=4,
            random_state=random_state,
        ),
    }

    if XGB_AVAILABLE:
        models["XGBoost Regressor"] = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
            eval_metric="rmse",
        )

    return models


def get_candidate_classifiers(random_state: int = 42) -> Dict[str, Any]:
    """Instantiates candidate classification models for academic risk detection."""
    models: Dict[str, Any] = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_split=4,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Gradient Boosting Classifier": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            min_samples_split=4,
            random_state=random_state,
        ),
    }

    if XGB_AVAILABLE:
        models["XGBoost Classifier"] = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
            eval_metric="logloss",
        )

    return models


def cross_validate_regressors(
    models: Dict[str, Any],
    X_train_trans: np.ndarray,
    y_train: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Evaluates regressors using 5-fold cross-validation on training data only."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results: Dict[str, Dict[str, float]] = {}

    for name, model in models.items():
        mae_scores: List[float] = []
        rmse_scores: List[float] = []
        r2_scores: List[float] = []

        for train_idx, val_idx in kf.split(X_train_trans):
            X_tr, X_val = X_train_trans[train_idx], X_train_trans[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            # Fit clone on fold
            model_clone = type(model)(**model.get_params())
            model_clone.fit(X_tr, y_tr)
            preds = model_clone.predict(X_val)

            metrics = compute_regression_metrics(y_val, preds)
            mae_scores.append(metrics["MAE"])
            rmse_scores.append(metrics["RMSE"])
            r2_scores.append(metrics["R2"])

        results[name] = {
            "MAE": round(float(np.mean(mae_scores)), 4),
            "MAE_std": round(float(np.std(mae_scores)), 4),
            "RMSE": round(float(np.mean(rmse_scores)), 4),
            "RMSE_std": round(float(np.std(rmse_scores)), 4),
            "R2": round(float(np.mean(r2_scores)), 4),
            "R2_std": round(float(np.std(r2_scores)), 4),
        }
        logger.info(f"CV Regressor [{name}]: RMSE={results[name]['RMSE']:.3f} (+/- {results[name]['RMSE_std']:.3f}), "
                    f"MAE={results[name]['MAE']:.3f}, R2={results[name]['R2']:.3f}")

    return results


def cross_validate_classifiers(
    models: Dict[str, Any],
    X_train_trans: np.ndarray,
    y_train: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Evaluates classifiers using 5-fold Stratified cross-validation on training data only."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results: Dict[str, Dict[str, float]] = {}

    for name, model in models.items():
        acc_scores: List[float] = []
        prec_scores: List[float] = []
        rec_scores: List[float] = []
        f1_scores: List[float] = []
        auc_scores: List[float] = []

        for train_idx, val_idx in skf.split(X_train_trans, y_train):
            X_tr, X_val = X_train_trans[train_idx], X_train_trans[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            model_clone = type(model)(**model.get_params())
            model_clone.fit(X_tr, y_tr)
            preds = model_clone.predict(X_val)

            probs = None
            if hasattr(model_clone, "predict_proba"):
                probs = model_clone.predict_proba(X_val)

            metrics = compute_classification_metrics(y_val, preds, probs)
            acc_scores.append(metrics["Accuracy"])
            prec_scores.append(metrics["Precision"])
            rec_scores.append(metrics["Recall"])
            f1_scores.append(metrics["F1_Score"])
            if metrics["ROC_AUC"] is not None:
                auc_scores.append(metrics["ROC_AUC"])

        results[name] = {
            "Accuracy": round(float(np.mean(acc_scores)), 4),
            "Precision": round(float(np.mean(prec_scores)), 4),
            "Recall": round(float(np.mean(rec_scores)), 4),
            "F1_Score": round(float(np.mean(f1_scores)), 4),
            "ROC_AUC": round(float(np.mean(auc_scores)), 4) if auc_scores else 0.0,
        }
        logger.info(f"CV Classifier [{name}]: F1={results[name]['F1_Score']:.3f}, "
                    f"Recall={results[name]['Recall']:.3f}, ROC-AUC={results[name]['ROC_AUC']:.3f}")

    return results


def run_training_pipeline(
    dataset_path: Optional[str | Path] = None,
    include_prior_grades: bool = False,
    models_dir: str | Path = "models",
    plots_dir: str | Path = "outputs/plots",
    metrics_dir: str | Path = "outputs/metrics",
    random_state: int = 42,
) -> Dict[str, Any]:
    """Runs the end-to-end ML pipeline with leak prevention, fair CV, and artifact serialization."""
    start_time = time.time()
    models_path = Path(models_dir)
    plots_path = Path(plots_dir)
    metrics_path = Path(metrics_dir)

    for p in [models_path, plots_path, metrics_path]:
        p.mkdir(parents=True, exist_ok=True)

    # 1. Load and Partition Dataset
    bundle = prepare_dataset_bundle(
        filepath=dataset_path,
        include_prior_grades=include_prior_grades,
        random_state=random_state,
    )

    # 2. Build and Fit Preprocessor strictly on training data
    preprocessor = build_preprocessor(bundle.categorical_cols, bundle.numeric_cols)
    preprocessor = fit_preprocessor(preprocessor, bundle.X_train)

    X_train_trans, feature_names = transform_data(preprocessor, bundle.X_train)
    X_test_trans, _ = transform_data(preprocessor, bundle.X_test)

    y_reg_tr = bundle.y_reg_train.to_numpy()
    y_reg_te = bundle.y_reg_test.to_numpy()
    y_clf_tr = bundle.y_clf_train.to_numpy()
    y_clf_te = bundle.y_clf_test.to_numpy()

    # -------------------------------------------------------------
    # TASK A: Student Grade Prediction (Regression)
    # -------------------------------------------------------------
    logger.info("=== Benchmarking Candidate Regressors ===")
    candidate_regressors = get_candidate_regressors(random_state=random_state)
    cv_reg_results = cross_validate_regressors(
        candidate_regressors,
        X_train_trans,
        y_reg_tr,
        n_splits=5,
        random_state=random_state,
    )

    # Model Selection: Pick regressor with lowest cross-validation RMSE
    best_reg_name = min(cv_reg_results.keys(), key=lambda k: cv_reg_results[k]["RMSE"])
    logger.info(f"Selected Best Regressor based on CV RMSE: {best_reg_name} (CV RMSE: {cv_reg_results[best_reg_name]['RMSE']})")

    # Fit all regressors on full training set to measure test set performance
    test_reg_results: Dict[str, Dict[str, float]] = {}
    fitted_regressors: Dict[str, Any] = {}

    for name, model in candidate_regressors.items():
        model.fit(X_train_trans, y_reg_tr)
        fitted_regressors[name] = model
        y_pred = model.predict(X_test_trans)
        test_reg_results[name] = compute_regression_metrics(y_reg_te, y_pred)
        logger.info(f"Test Set Evaluation [{name}]: RMSE={test_reg_results[name]['RMSE']}, "
                    f"MAE={test_reg_results[name]['MAE']}, R2={test_reg_results[name]['R2']}")

    best_reg_model = fitted_regressors[best_reg_name]
    best_reg_test_preds = best_reg_model.predict(X_test_trans)

    # Generate Regression Diagnostic Visualizations
    plot_actual_vs_predicted(
        y_true=y_reg_te,
        y_pred=best_reg_test_preds,
        model_name=best_reg_name,
        output_path=plots_path / "regression_actual_vs_predicted.png",
    )
    plot_residuals(
        y_true=y_reg_te,
        y_pred=best_reg_test_preds,
        model_name=best_reg_name,
        output_path=plots_path / "regression_residuals.png",
    )
    plot_regression_model_comparison(
        cv_results=cv_reg_results,
        test_results=test_reg_results,
        output_path=plots_path / "regression_model_comparison.png",
    )

    # -------------------------------------------------------------
    # TASK B: Academic Risk Classification
    # -------------------------------------------------------------
    logger.info("=== Benchmarking Candidate Risk Classifiers ===")
    candidate_classifiers = get_candidate_classifiers(random_state=random_state)
    cv_clf_results = cross_validate_classifiers(
        candidate_classifiers,
        X_train_trans,
        y_clf_tr,
        n_splits=5,
        random_state=random_state,
    )

    # Model Selection: Pick classifier with highest cross-validation F1-score / ROC-AUC
    # We combine F1 and ROC_AUC for robust early-warning performance
    best_clf_name = max(cv_clf_results.keys(), key=lambda k: (cv_clf_results[k]["F1_Score"] + cv_clf_results[k]["ROC_AUC"]))
    logger.info(f"Selected Best Classifier based on CV: {best_clf_name} (CV F1: {cv_clf_results[best_clf_name]['F1_Score']}, AUC: {cv_clf_results[best_clf_name]['ROC_AUC']})")

    test_clf_results: Dict[str, Dict[str, Any]] = {}
    fitted_classifiers: Dict[str, Any] = {}
    roc_plot_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}
    calibration_plot_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}

    for name, model in candidate_classifiers.items():
        model.fit(X_train_trans, y_clf_tr)
        fitted_classifiers[name] = model
        y_pred_clf = model.predict(X_test_trans)
        y_prob_clf = model.predict_proba(X_test_trans) if hasattr(model, "predict_proba") else None

        test_clf_results[name] = compute_classification_metrics(y_clf_te, y_pred_clf, y_prob_clf)
        logger.info(f"Test Set Evaluation [{name}]: Accuracy={test_clf_results[name]['Accuracy']}, "
                    f"F1={test_clf_results[name]['F1_Score']}, Recall={test_clf_results[name]['Recall']}, "
                    f"ROC-AUC={test_clf_results[name]['ROC_AUC']}, Brier={test_clf_results[name]['Brier_Score']}")

        if y_prob_clf is not None:
            from sklearn.calibration import calibration_curve
            from sklearn.metrics import auc, roc_curve
            fpr, tpr, _ = roc_curve(y_clf_te, y_prob_clf[:, 1])
            roc_plot_data[name] = (fpr, tpr, float(auc(fpr, tpr)))

            prob_true, prob_pred = calibration_curve(y_clf_te, y_prob_clf[:, 1], n_bins=5)
            calibration_plot_data[name] = (prob_true, prob_pred, test_clf_results[name]["Brier_Score"])

    best_clf_model = fitted_classifiers[best_clf_name]
    best_clf_cm = test_clf_results[best_clf_name]["Confusion_Matrix"]

    # Generate Classification Diagnostic Visualizations
    plot_confusion_matrix_heatmap(
        cm=best_clf_cm,
        model_name=best_clf_name,
        output_path=plots_path / "classification_confusion_matrix.png",
    )
    if roc_plot_data:
        plot_roc_curves(
            roc_data=roc_plot_data,
            output_path=plots_path / "classification_roc_curve.png",
        )
    if calibration_plot_data:
        plot_calibration_curves(
            calibration_data=calibration_plot_data,
            output_path=plots_path / "classification_calibration_curves.png",
        )
    plot_classification_comparison(
        cv_results=cv_clf_results,
        test_results=test_clf_results,
        output_path=plots_path / "classification_model_comparison.png",
    )

    # -------------------------------------------------------------
    # TASK C: Model Explainability & SHAP Attribution
    # -------------------------------------------------------------
    logger.info("=== Generating Explainability and Feature Attribution Report ===")
    explainer, df_shap = generate_shap_report(
        model=best_reg_model,
        X_train_trans=X_train_trans,
        X_test_trans=X_test_trans,
        feature_names=feature_names,
        output_plot_path=plots_path / "feature_importance_shap.png",
    )

    # -------------------------------------------------------------
    # TASK D: Artifact Serialization & Metadata
    # -------------------------------------------------------------
    logger.info("=== Serializing Production Models and Pipelines ===")

    # 1. Save Preprocessor
    preprocessor_file = models_path / "preprocessor.joblib"
    save_preprocessor(preprocessor, preprocessor_file)

    # 2. Save Individual Selected Estimators
    reg_file = models_path / "student_grade_regressor.joblib"
    clf_file = models_path / "student_risk_classifier.joblib"
    joblib.dump(best_reg_model, reg_file)
    joblib.dump(best_clf_model, clf_file)

    # 3. Save Full Self-Contained Scikit-Learn Pipelines
    full_reg_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", best_reg_model),
    ])
    full_clf_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", best_clf_model),
    ])
    joblib.dump(full_reg_pipeline, models_path / "full_regression_pipeline.joblib")
    joblib.dump(full_clf_pipeline, models_path / "full_classification_pipeline.joblib")

    # 4. Save Comprehensive Model Metadata
    metadata = {
        "project": "EduPredict AI",
        "timestamp": datetime.now().isoformat(),
        "training_duration_seconds": round(time.time() - start_time, 2),
        "scenario": bundle.scenario,
        "include_prior_grades": include_prior_grades,
        "dataset_info": bundle.metadata,
        "features": {
            "raw_feature_count": len(bundle.feature_names),
            "raw_features": bundle.feature_names,
            "categorical_features": bundle.categorical_cols,
            "numeric_features": bundle.numeric_cols,
            "transformed_feature_count": len(feature_names),
            "transformed_features": feature_names,
        },
        "selected_models": {
            "regression": {
                "algorithm": best_reg_name,
                "hyperparameters": {k: str(v) for k, v in best_reg_model.get_params().items()},
                "cross_validation_5fold": cv_reg_results[best_reg_name],
                "held_out_test_metrics": test_reg_results[best_reg_name],
            },
            "classification": {
                "algorithm": best_clf_name,
                "hyperparameters": {k: str(v) for k, v in best_clf_model.get_params().items()},
                "cross_validation_5fold": cv_clf_results[best_clf_name],
                "held_out_test_metrics": test_clf_results[best_clf_name],
            },
        },
        "all_benchmarked_models": {
            "regression_cv": cv_reg_results,
            "regression_test": test_reg_results,
            "classification_cv": cv_clf_results,
            "classification_test": test_clf_results,
        },
        "academic_context_and_limitations": {
            "prediction_scenario": bundle.scenario,
            "rationale_for_performance": "Prior grades G1 and G2 are excluded to eliminate data leakage and permit true early intervention. In educational data mining benchmarks (Cortez & Silva 2008), pre-enrollment background variables account for approx 25-30% of grade variance (R2 ~ 0.26-0.28, MAE ~ 2.90), as human examination grades depend on unobserved term dynamics.",
            "decision_threshold_policy": "Default risk classification threshold is 0.50, configurable in EduPredictor. Predicted risks are indicators for supportive mentoring, not definitive performance labels.",
        },
    }

    metadata_path = models_path / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Save metrics summaries in outputs/metrics/
    with open(metrics_path / "regression_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"cv_5fold": cv_reg_results, "test_evaluation": test_reg_results}, f, indent=2)

    with open(metrics_path / "classification_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"cv_5fold": cv_clf_results, "test_evaluation": test_clf_results}, f, indent=2)

    logger.info(f"Pipeline complete! Artifacts saved to: {models_path.resolve()}")
    return metadata


if __name__ == "__main__":
    meta = run_training_pipeline()
    print("\n" + "="*60)
    print("EDUPREDICT AI - TRAINING PIPELINE EXECUTION SUMMARY")
    print("="*60)
    print(f"Scenario: {meta['scenario']}")
    print(f"Selected Regressor: {meta['selected_models']['regression']['algorithm']}")
    print(f"Test RMSE: {meta['selected_models']['regression']['held_out_test_metrics']['RMSE']}")
    print(f"Test MAE:  {meta['selected_models']['regression']['held_out_test_metrics']['MAE']}")
    print(f"Test R2:   {meta['selected_models']['regression']['held_out_test_metrics']['R2']}")
    print("-" * 60)
    print(f"Selected Classifier: {meta['selected_models']['classification']['algorithm']}")
    print(f"Test Accuracy: {meta['selected_models']['classification']['held_out_test_metrics']['Accuracy']}")
    print(f"Test F1:       {meta['selected_models']['classification']['held_out_test_metrics']['F1_Score']}")
    print(f"Test Recall:   {meta['selected_models']['classification']['held_out_test_metrics']['Recall']}")
    print(f"Test ROC-AUC:  {meta['selected_models']['classification']['held_out_test_metrics']['ROC_AUC']}")
    print("="*60)
