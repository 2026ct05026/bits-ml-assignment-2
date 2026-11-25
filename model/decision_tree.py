"""Decision Tree classifier module."""

from __future__ import annotations

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from model.config import RANDOM_STATE
from model.evaluation import EvaluationResult, build_model_pipeline, train_evaluate_and_save
from model.preprocessing import prepare_train_test_data

MODEL_DISPLAY_NAME = "Decision Tree"
ARTIFACT_FILENAME = "decision_tree_model.joblib"


def build_pipeline():
    """Create the preprocessing + Decision Tree pipeline (scaling not required for trees)."""
    estimator = DecisionTreeClassifier(
        criterion="gini",
        max_depth=8,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
    )
    return build_model_pipeline(estimator, scale_numeric=False)


def run_decision_tree(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> EvaluationResult:
    """Train, evaluate, and persist the Decision Tree model."""
    return train_evaluate_and_save(
        model_name=MODEL_DISPLAY_NAME,
        pipeline=build_pipeline(),
        artifact_filename=ARTIFACT_FILENAME,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, _ = prepare_train_test_data(save_test_csv=False)
    result = run_decision_tree(X_tr, X_te, y_tr, y_te)
    print(result.to_row())
