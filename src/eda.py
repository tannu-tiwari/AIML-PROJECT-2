"""Exploratory Data Analysis (EDA) Module for EduPredict AI.

Generates academic-quality visualizations and statistical summaries of student performance,
study habits, behavioral variables, and failure risk factors.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data_loader import load_raw_dataset, validate_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
PALETTE = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]


def ensure_output_dirs(output_dir: Path) -> Path:
    """Ensures output directory exists."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def plot_target_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """Plots histogram and KDE for final grade G3, highlighting pass/fail threshold."""
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    sns.histplot(
        df["G3"],
        kde=True,
        bins=21,
        color="#1f77b4",
        edgecolor="black",
        ax=ax,
        stat="density",
    )

    mean_val = df["G3"].mean()
    median_val = df["G3"].median()

    ax.axvline(10, color="#d62728", linestyle="--", linewidth=2, label="Pass Threshold (10/20)")
    ax.axvline(mean_val, color="#2ca02c", linestyle="-", linewidth=2, label=f"Mean ({mean_val:.2f})")
    ax.axvline(median_val, color="#ff7f0e", linestyle=":", linewidth=2, label=f"Median ({median_val:.1f})")

    ax.set_title("Distribution of Final Academic Grades (G3)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Final Grade (0 - 20 Scale)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_xlim(-0.5, 20.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved target grade distribution plot to: {output_path}")


def plot_data_quality_summary(df: pd.DataFrame, output_path: Path) -> None:
    """Generates visual audit showing missing values and duplicate records."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), dpi=300)

    missing_series = df.isnull().sum()
    top_missing = missing_series[missing_series > 0]
    if len(top_missing) == 0:
        # Zero missing values confirmed
        ax1.bar(["Total Missing Values"], [0], color="#2ca02c", width=0.4, edgecolor="black")
        ax1.set_ylim(0, 10)
        ax1.text(0, 0.5, "0 Missing Values\n(100% Complete)", ha="center", va="bottom", fontsize=11, fontweight="bold", color="#2ca02c")
    else:
        top_missing.plot(kind="bar", ax=ax1, color="#d62728")

    ax1.set_title("Missing Values Audit", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Missing Count")

    dup_count = int(df.duplicated().sum())
    ax2.bar(["Duplicate Records"], [dup_count], color="#1f77b4", width=0.4, edgecolor="black")
    ax2.set_ylim(0, 10)
    ax2.text(0, 0.5, f"{dup_count} Duplicates Found\n(Unique Rows: {len(df)})", ha="center", va="bottom", fontsize=11, fontweight="bold", color="#1f77b4")
    ax2.set_title("Duplicate Records Audit", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Duplicate Count")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved data quality summary plot to: {output_path}")


def plot_correlation_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    """Plots Pearson correlation heatmap for all numeric features and targets."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient"},
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        ax=ax,
    )
    ax.set_title("Correlation Matrix of Numeric Attributes & Exam Outcomes", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved correlation heatmap to: {output_path}")


def plot_studytime_and_absences(df: pd.DataFrame, output_path: Path) -> None:
    """Analyzes impact of weekly study time, class absences, and past failures on G3."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)

    # 1. Study time vs G3
    studytime_labels = {1: "< 2 hrs", 2: "2 - 5 hrs", 3: "5 - 10 hrs", 4: "> 10 hrs"}
    df_plot = df.copy()
    df_plot["studytime_label"] = df_plot["studytime"].map(studytime_labels)
    study_order = ["< 2 hrs", "2 - 5 hrs", "5 - 10 hrs", "> 10 hrs"]

    sns.boxplot(
        data=df_plot,
        x="studytime_label",
        y="G3",
        order=study_order,
        hue="studytime_label",
        palette="Blues",
        legend=False,
        ax=axes[0],
    )
    axes[0].axhline(10, color="red", linestyle="--", alpha=0.7)
    axes[0].set_title("Final Grade by Weekly Study Time", fontweight="bold", fontsize=11)
    axes[0].set_xlabel("Weekly Study Time")
    axes[0].set_ylabel("Final Grade (G3)")

    # 2. Past Failures vs G3
    sns.boxplot(
        data=df,
        x="failures",
        y="G3",
        hue="failures",
        palette="Reds",
        legend=False,
        ax=axes[1],
    )
    axes[1].axhline(10, color="red", linestyle="--", alpha=0.7)
    axes[1].set_title("Final Grade by Past Academic Failures", fontweight="bold", fontsize=11)
    axes[1].set_xlabel("Number of Past Failures")
    axes[1].set_ylabel("Final Grade (G3)")

    # 3. Absences vs G3 scatter plot with trend
    sns.regplot(
        data=df,
        x="absences",
        y="G3",
        scatter_kws={"alpha": 0.5, "color": "#1f77b4"},
        line_kws={"color": "#d62728", "linewidth": 2},
        ax=axes[2],
    )
    axes[2].axhline(10, color="gray", linestyle="--", alpha=0.5)
    axes[2].set_title("Absences vs. Final Grade (with Trend)", fontweight="bold", fontsize=11)
    axes[2].set_xlabel("Number of School Absences")
    axes[2].set_ylabel("Final Grade (G3)")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved studytime and absences analysis to: {output_path}")


def plot_academic_risk_breakdown(df: pd.DataFrame, output_path: Path) -> None:
    """Visualizes class balance of Academic Risk (G3 < 10 vs G3 >= 10)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    risk_series = (df["G3"] < 10).map({True: "At Risk (Fail < 10)", False: "Passing (Grade >= 10)"})
    counts = risk_series.value_counts()

    # Bar plot
    colors = ["#2ca02c", "#d62728"]
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette=colors, legend=False, ax=ax1, edgecolor="black")
    ax1.set_title("Academic Risk Distribution (Count)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Number of Students")
    for i, v in enumerate(counts.values):
        ax1.text(i, v + 4, f"{v} ({v/len(df):.1%})", ha="center", fontweight="bold", fontsize=10)

    # Pie chart
    ax2.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        explode=(0.04, 0.04),
        textprops={"fontsize": 11, "fontweight": "bold"},
    )
    ax2.set_title("Academic Risk Proportion", fontsize=12, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved academic risk breakdown to: {output_path}")


def plot_socioeconomic_factors(df: pd.DataFrame, output_path: Path) -> None:
    """Examines parental education, school support, and higher education aspirations."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)

    # 1. Mother's Education vs G3
    edu_labels = {0: "None", 1: "4th Gr", 2: "5-9th Gr", 3: "Secondary", 4: "Higher"}
    df_plot = df.copy()
    df_plot["Medu_label"] = df_plot["Medu"].map(edu_labels)
    order = ["None", "4th Gr", "5-9th Gr", "Secondary", "Higher"]

    sns.barplot(
        data=df_plot,
        x="Medu_label",
        y="G3",
        order=order,
        hue="Medu_label",
        palette="Purples",
        legend=False,
        ax=axes[0],
        errorbar=None,
        edgecolor="black",
    )
    axes[0].set_title("Mean Grade by Mother's Education", fontweight="bold", fontsize=11)
    axes[0].set_xlabel("Mother's Education Level")
    axes[0].set_ylabel("Average G3 Grade")

    # 2. Desires Higher Education vs G3
    sns.boxplot(
        data=df,
        x="higher",
        y="G3",
        hue="higher",
        palette={"yes": "#2ca02c", "no": "#d62728"},
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("Higher Education Ambition vs. Grade", fontweight="bold", fontsize=11)
    axes[1].set_xlabel("Desires Higher Education (higher)")
    axes[1].set_ylabel("Final Grade (G3)")

    # 3. Extra School Support vs G3
    sns.boxplot(
        data=df,
        x="schoolsup",
        y="G3",
        hue="schoolsup",
        palette={"yes": "#ff7f0e", "no": "#1f77b4"},
        legend=False,
        ax=axes[2],
    )
    axes[2].set_title("School Support (Remedial) vs. Grade", fontweight="bold", fontsize=11)
    axes[2].set_xlabel("Extra School Support (schoolsup)")
    axes[2].set_ylabel("Final Grade (G3)")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved socioeconomic factors plot to: {output_path}")


def run_full_eda(
    dataset_path: Optional[str | Path] = None,
    plots_dir: Optional[str | Path] = None,
    metrics_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Executes full exploratory analysis and exports charts & summary statistics."""
    df = load_raw_dataset(dataset_path)

    plots_path = Path(plots_dir or "outputs/plots")
    metrics_path = Path(metrics_dir or "outputs/metrics")
    ensure_output_dirs(plots_path)
    ensure_output_dirs(metrics_path)

    validation_info = validate_dataset(df)

    # Generate all requested visualizations
    plot_target_distribution(df, plots_path / "target_grade_distribution.png")
    plot_data_quality_summary(df, plots_path / "missing_and_duplicates.png")
    plot_correlation_heatmap(df, plots_path / "correlation_heatmap.png")
    plot_studytime_and_absences(df, plots_path / "studytime_absences_analysis.png")
    plot_academic_risk_breakdown(df, plots_path / "academic_risk_distribution.png")
    plot_socioeconomic_factors(df, plots_path / "socioeconomic_factors.png")

    # Generate statistical summary
    eda_summary = {
        "dataset_name": "UCI Student Performance (Mathematics)",
        "total_records": len(df),
        "total_attributes": len(df.columns),
        "target_variable": "G3",
        "target_stats": {
            "mean": round(float(df["G3"].mean()), 3),
            "std": round(float(df["G3"].std()), 3),
            "median": round(float(df["G3"].median()), 3),
            "iqr": round(float(df["G3"].quantile(0.75) - df["G3"].quantile(0.25)), 3),
            "min": int(df["G3"].min()),
            "max": int(df["G3"].max()),
        },
        "risk_classification_stats": {
            "pass_count (G3>=10)": int((df["G3"] >= 10).sum()),
            "fail_count (G3<10)": int((df["G3"] < 10).sum()),
            "fail_rate": round(float((df["G3"] < 10).mean()), 4),
        },
        "top_correlations_with_G3": {
            k: round(float(v), 3)
            for k, v in df.select_dtypes(include=[np.number]).corr()["G3"].drop("G3").sort_values(ascending=False).items()
        },
        "data_quality": validation_info,
    }

    summary_file = metrics_path / "eda_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(eda_summary, f, indent=2)

    logger.info(f"Exported EDA summary metrics to: {summary_file}")
    return eda_summary


if __name__ == "__main__":
    summary = run_full_eda()
    print("EDA Complete. Key Summary:")
    print(json.dumps(summary, indent=2))
