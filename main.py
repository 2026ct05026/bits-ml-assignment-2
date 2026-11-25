"""End-to-end execution script for BITS Pilani M.Tech (AIML/DSE) ML Assignment."""

from __future__ import annotations

from model.config import RANDOM_STATE, TEST_DATA_PATH, TEST_SIZE
from model.decision_tree import run_decision_tree
from model.evaluation import generate_visualizations, save_metrics_table
from model.knn import run_knn
from model.logistic_regression import run_logistic_regression
from model.naive_bayes import run_naive_bayes
from model.preprocessing import prepare_train_test_data
from model.random_forest import run_random_forest


def print_dataset_summary(summary) -> None:
    """Display dataset cleaning, missing-value, and split statistics."""
    print("=" * 72)
    print("UCI THYROID DISEASE ('sick') CLASSIFICATION PIPELINE")
    print("=" * 72)
    print(f"Raw dataset shape              : {summary.raw_shape}")
    print(f"Columns with missing values    : {len(summary.missing_counts_raw)}")
    for col, cnt in summary.missing_counts_raw.items():
        print(f"  - {col:<24}: {cnt} missing")
    print(f"Invalid age values fixed (>120): {summary.invalid_age_fixed}")
    print(f"Exact duplicate rows removed   : {summary.duplicate_rows_removed}")
    print(f"Cleaned dataset shape          : {summary.cleaned_shape}")
    print(f"Cleaned class distribution     : {summary.class_counts_cleaned}")
    print(
        f"Stratified split ({int((1 - TEST_SIZE) * 100)}:{int(TEST_SIZE * 100)}, seed={RANDOM_STATE})"
    )
    print(f"  - Train shape: {summary.train_shape} | counts: {summary.train_class_counts}")
    print(f"  - Test shape : {summary.test_shape} | counts: {summary.test_class_counts}")
    print(f"Saved test split CSV to        : {TEST_DATA_PATH}")
    print("-" * 72)


def main() -> None:
    """Run preprocessing, train all 5 models, save artifacts, and generate metrics/plots."""
    X_train, X_test, y_train, y_test, summary = prepare_train_test_data(save_test_csv=True)
    print_dataset_summary(summary)

    runners = [
        run_logistic_regression,
        run_decision_tree,
        run_knn,
        run_naive_bayes,
        run_random_forest,
    ]

    results = [runner(X_train, X_test, y_train, y_test) for runner in runners]

    metrics_df = save_metrics_table(results)
    generate_visualizations(results, y_test, summary)

    print("\nMODEL EVALUATION COMPARISON (TEST SET)")
    print("-" * 72)
    print(metrics_df.to_string(index=False))
    print("-" * 72)
    print("Saved trained model artifacts:")
    for res in results:
        print(f"  - {res.model_name:<26} -> {res.artifact_path}")
    print("\nVisualizations saved in ./results/ (class_distribution.png, metrics_comparison.png, roc_curves.png, confusion_matrices.png)")


if __name__ == "__main__":
    main()
