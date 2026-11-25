"""Shared training, evaluation metrics, artifact persistence, and chart generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from model.config import MODEL_DIR, RESULTS_DIR
from model.preprocessing import DatasetSummary, build_preprocessor


@dataclass
class EvaluationResult:
    """Stores computed classification metrics and test predictions for one model."""

    model_name: str
    accuracy: float
    auc: float
    precision: float
    recall: float
    f1: float
    mcc: float
    y_pred: np.ndarray
    y_prob: np.ndarray
    confusion: np.ndarray
    artifact_path: Path

    def to_row(self) -> dict[str, object]:
        """Return metrics formatted as a flat table row."""
        return {
            "ML Model Name": self.model_name,
            "Accuracy": round(self.accuracy, 4),
            "AUC": round(self.auc, 4),
            "Precision": round(self.precision, 4),
            "Recall": round(self.recall, 4),
            "F1": round(self.f1, 4),
            "MCC": round(self.mcc, 4),
        }


def build_model_pipeline(
    estimator: BaseEstimator, scale_numeric: bool = True
) -> Pipeline:
    """Wrap preprocessing and estimator into a single leak-free Pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=scale_numeric)),
            ("classifier", estimator),
        ]
    )


def train_evaluate_and_save(
    model_name: str,
    pipeline: Pipeline,
    artifact_filename: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> EvaluationResult:
    """Fit pipeline on training data only, evaluate on test split, and save artifact."""
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
    else:
        y_prob = pipeline.decision_function(X_test)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = MODEL_DIR / artifact_filename
    joblib.dump(pipeline, artifact_path)

    return EvaluationResult(
        model_name=model_name,
        accuracy=float(accuracy_score(y_test, y_pred)),
        auc=float(roc_auc_score(y_test, y_prob)),
        precision=float(precision_score(y_test, y_pred, zero_division=0)),
        recall=float(recall_score(y_test, y_pred, zero_division=0)),
        f1=float(f1_score(y_test, y_pred, zero_division=0)),
        mcc=float(matthews_corrcoef(y_test, y_pred)),
        y_pred=y_pred,
        y_prob=y_prob,
        confusion=confusion_matrix(y_test, y_pred),
        artifact_path=artifact_path,
    )


def save_metrics_table(results: list[EvaluationResult]) -> pd.DataFrame:
    """Persist comparison metrics as a CSV file and return the DataFrame."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([r.to_row() for r in results])
    df.to_csv(RESULTS_DIR / "metrics_comparison.csv", index=False)
    return df


def generate_visualizations(
    results: list[EvaluationResult],
    y_test: pd.Series,
    summary: DatasetSummary,
) -> None:
    """Create publication-ready plots for class distribution, metrics, ROC, and confusion matrices."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    # 1. Class distribution plot (overall cleaned dataset vs train/test splits)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    overall_labels = ["negative (0)", "sick (1)"]
    overall_counts = [
        summary.class_counts_cleaned.get("negative", 0),
        summary.class_counts_cleaned.get("sick", 0),
    ]
    sns.barplot(
        x=overall_labels,
        y=overall_counts,
        hue=overall_labels,
        palette=["#4C72B0", "#DD8452"],
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title("Cleaned Dataset Class Distribution (n=3,711)")
    axes[0].set_ylabel("Instance Count")
    for idx, count in enumerate(overall_counts):
        pct = 100.0 * count / sum(overall_counts)
        axes[0].text(idx, count + 50, f"{count} ({pct:.2f}%)", ha="center", fontsize=9)

    split_df = pd.DataFrame(
        {
            "Split": ["Train (80%)", "Train (80%)", "Test (20%)", "Test (20%)"],
            "Class": ["negative (0)", "sick (1)", "negative (0)", "sick (1)"],
            "Proportion (%)": [
                100.0 * summary.train_class_counts[0] / sum(summary.train_class_counts.values()),
                100.0 * summary.train_class_counts[1] / sum(summary.train_class_counts.values()),
                100.0 * summary.test_class_counts[0] / sum(summary.test_class_counts.values()),
                100.0 * summary.test_class_counts[1] / sum(summary.test_class_counts.values()),
            ],
        }
    )
    sns.barplot(
        data=split_df,
        x="Split",
        y="Proportion (%)",
        hue="Class",
        palette=["#4C72B0", "#DD8452"],
        ax=axes[1],
    )
    axes[1].set_title("Stratified Split Preserves Class Imbalance")
    axes[1].set_ylim(0, 105)
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.2f%%", fontsize=8, padding=2)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "class_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 2. All 6 evaluation metrics comparison bar chart
    metrics_df = pd.DataFrame([r.to_row() for r in results])
    melted = metrics_df.melt(
        id_vars="ML Model Name", var_name="Metric", value_name="Score"
    )
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.barplot(
        data=melted,
        x="ML Model Name",
        y="Score",
        hue="Metric",
        palette="viridis",
        ax=ax,
    )
    ax.set_title("Model Performance Comparison Across All Six Evaluation Metrics", fontsize=13, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.12)
    ax.legend(title="Metric", bbox_to_anchor=(1.01, 1), loc="upper left")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=7.5, rotation=90, padding=3)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "metrics_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 3. ROC curves across all 5 models
    fig, ax = plt.subplots(figsize=(8, 6))
    for res in results:
        fpr, tpr, _ = roc_curve(y_test, res.y_prob)
        ax.plot(fpr, tpr, lw=2, label=f"{res.model_name} (AUC = {res.auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Baseline (AUC = 0.5000)")
    ax.set_title("ROC Curves Comparison Across All Five Classifiers", fontsize=13)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "roc_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 4. Confusion matrices grid
    fig, axes = plt.subplots(1, len(results), figsize=(20, 4))
    for ax, res in zip(axes, results):
        sns.heatmap(
            res.confusion,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Neg", "Sick"],
            yticklabels=["Neg", "Sick"],
            ax=ax,
        )
        ax.set_title(res.model_name, fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.suptitle("Confusion Matrices on 20% Stratified Test Set (n=743)", fontsize=13, y=1.04)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrices.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
