"""Model evaluation and performance visualization module for EduPredict AI.

Computes academic-standard regression and classification metrics,
generates publication-ready diagnostic charts, and exports JSON reports.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_curve,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculates MAE, MSE, RMSE, R2, and Max Error for regression predictions."""
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    max_err = float(np.max(np.abs(y_true - y_pred)))

    return {
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "Max_Error": round(max_err, 4),
    }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Calculates Accuracy, Precision, Recall, F1, ROC-AUC, Brier score and Confusion Matrix."""
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    roc_auc_val: Optional[float] = None
    brier_val: Optional[float] = None
    if y_prob is not None:
        try:
            # Check if 2D or 1D probability
            prob_pos = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
            fpr, tpr, _ = roc_curve(y_true, prob_pos)
            roc_auc_val = float(auc(fpr, tpr))
            brier_val = float(brier_score_loss(y_true, prob_pos))
        except Exception as e:
            logger.warning(f"Failed to compute ROC-AUC or Brier score: {e}")
            roc_auc_val = None
            brier_val = None

    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1_Score": round(f1, 4),
        "ROC_AUC": round(roc_auc_val, 4) if roc_auc_val is not None else None,
        "Brier_Score": round(brier_val, 4) if brier_val is not None else None,
        "Confusion_Matrix": cm,
    }


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_path: Path,
) -> None:
    """Plots actual vs. predicted student grades with 45-degree reference line."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    ax.scatter(y_true, y_pred, alpha=0.7, color="#1f77b4", edgecolor="black", s=50, label="Student Predictions")

    # Ideal diagonal
    min_val = min(float(y_true.min()), float(y_pred.min()), 0)
    max_val = max(float(y_true.max()), float(y_pred.max()), 20)
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Ideal 1:1 Parity")

    # Metrics annotation box
    metrics = compute_regression_metrics(y_true, y_pred)
    text_str = f"RMSE: {metrics['RMSE']:.2f}\nMAE: {metrics['MAE']:.2f}\nR²: {metrics['R2']:.2f}"
    props = dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.5)
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=11, verticalalignment="top", bbox=props)

    ax.set_title(f"Actual vs. Predicted Final Grades - {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Actual Grade (G3: 0 - 20)", fontsize=11)
    ax.set_ylabel("Predicted Grade (G3: 0 - 20)", fontsize=11)
    ax.set_xlim(min_val - 1, max_val + 1)
    ax.set_ylim(min_val - 1, max_val + 1)
    ax.legend(loc="lower right")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved actual vs. predicted plot to: {output_path}")


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_path: Path,
) -> None:
    """Visualizes residual distribution and residual vs. fitted values."""
    residuals = y_true - y_pred

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # 1. Residuals vs Predicted
    ax1.scatter(y_pred, residuals, alpha=0.7, color="#2b5c8f", edgecolor="black")
    ax1.axhline(0, color="red", linestyle="--", linewidth=2)
    ax1.set_title(f"Residuals vs. Fitted Values ({model_name})", fontweight="bold", fontsize=11)
    ax1.set_xlabel("Predicted Grade (G3)")
    ax1.set_ylabel("Residual (Actual - Predicted)")

    # 2. Residual Distribution
    sns.histplot(residuals, kde=True, ax=ax2, color="#7570b3", edgecolor="black")
    ax2.axvline(0, color="red", linestyle="--", linewidth=2)
    ax2.set_title(f"Residual Distribution ({model_name})", fontweight="bold", fontsize=11)
    ax2.set_xlabel("Residual (Grade Points)")
    ax2.set_ylabel("Count")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved residual plot to: {output_path}")


def plot_regression_model_comparison(
    cv_results: Dict[str, Dict[str, float]],
    test_results: Dict[str, Dict[str, float]],
    output_path: Path,
) -> None:
    """Generates comparative bar chart of MAE, RMSE, and R2 across regressors."""
    models = list(cv_results.keys())
    metrics_to_plot = ["MAE", "RMSE", "R2"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)

    for i, metric in enumerate(metrics_to_plot):
        cv_vals = [cv_results[m].get(f"cv_{metric.lower()}_mean", cv_results[m].get(metric, 0)) for m in models]
        test_vals = [test_results[m][metric] for m in models]

        x = np.arange(len(models))
        width = 0.35

        rects1 = axes[i].bar(x - width/2, cv_vals, width, label="CV (5-Fold Train)", color="#2b5c8f", edgecolor="black")
        rects2 = axes[i].bar(x + width/2, test_vals, width, label="Held-Out Test", color="#e7298a", edgecolor="black")

        axes[i].set_title(f"Model Comparison: {metric}", fontweight="bold", fontsize=12)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(models, rotation=25, ha="right", fontsize=9)
        axes[i].set_ylabel(metric)
        axes[i].legend()

        # Add value labels
        for rect in rects1:
            h = rect.get_height()
            axes[i].annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width()/2, h),
                             xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
        for rect in rects2:
            h = rect.get_height()
            axes[i].annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width()/2, h),
                             xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved regression comparison plot to: {output_path}")


def plot_confusion_matrix_heatmap(
    cm: List[List[int]],
    model_name: str,
    output_path: Path,
) -> None:
    """Generates labeled confusion matrix heatmap with counts and percentages."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

    cm_arr = np.array(cm)
    cm_norm = cm_arr.astype("float") / cm_arr.sum(axis=1)[:, np.newaxis]

    labels = [
        [f"{cm_arr[i, j]}\n({cm_norm[i, j]:.1%})" for j in range(cm_arr.shape[1])]
        for i in range(cm_arr.shape[0])
    ]

    sns.heatmap(
        cm_arr,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pass (0)", "At Risk (1)"],
        yticklabels=["Pass (0)", "At Risk (1)"],
        ax=ax,
        annot_kws={"fontsize": 11, "fontweight": "bold"},
    )

    ax.set_title(f"Confusion Matrix - {model_name}\n(Academic Risk: G3 < 10)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=10, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=10, fontweight="bold")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved confusion matrix heatmap to: {output_path}")


def plot_roc_curves(
    roc_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    output_path: Path,
) -> None:
    """Plots comparative ROC curves with AUC annotations."""
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for idx, (name, (fpr, tpr, roc_auc)) in enumerate(roc_data.items()):
        color = colors[idx % len(colors)]
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Chance (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=10)
    ax.set_title("Receiver Operating Characteristic (ROC) - Risk Models", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved ROC curves plot to: {output_path}")


def plot_classification_comparison(
    cv_results: Dict[str, Dict[str, float]],
    test_results: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> None:
    """Generates multi-metric comparison for classification models."""
    models = list(cv_results.keys())
    metrics = ["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    x = np.arange(len(models))
    width = 0.15

    for idx, metric in enumerate(metrics):
        vals = [test_results[m].get(metric, 0.0) or 0.0 for m in models]
        rects = ax.bar(x + (idx - 2) * width, vals, width, label=metric, edgecolor="black")
        for rect in rects:
            h = rect.get_height()
            if h > 0:
                ax.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width()/2, h),
                            xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7)

    ax.set_title("Classification Model Performance Comparison (Test Set)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", ncol=len(metrics))

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved classification comparison plot to: {output_path}")


def plot_calibration_curves(
    calibration_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    output_path: Path,
) -> None:
    """Generates reliability calibration diagram for risk classification models."""
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for idx, (name, (prob_true, prob_pred, brier)) in enumerate(calibration_data.items()):
        color = colors[idx % len(colors)]
        brier_text = f" (Brier={brier:.3f})" if brier is not None else ""
        ax.plot(prob_pred, prob_true, marker="o", linewidth=2, color=color, label=f"{name}{brier_text}")

    # Perfectly calibrated line
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    ax.set_xlabel("Mean Predicted Probability", fontsize=10)
    ax.set_ylabel("Fraction of True Positive Cases", fontsize=10)
    ax.set_title("Probability Calibration Diagram (Reliability Curve)", fontsize=12, fontweight="bold")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved probability calibration diagram to: {output_path}")
