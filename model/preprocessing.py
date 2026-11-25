"""Dataset loading, cleaning, stratified splitting, and leak-free preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from model.config import (
    CATEGORICAL_FEATURES,
    DATA_DIR,
    DROP_COLUMNS,
    NEGATIVE_LABEL,
    NUMERIC_FEATURES,
    OPENML_DATASET_ID,
    POSITIVE_LABEL,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    TEST_DATA_PATH,
    TEST_SIZE,
)


@dataclass
class DatasetSummary:
    """Metadata captured during dataset inspection and cleaning."""

    raw_shape: tuple[int, int]
    missing_counts_raw: dict[str, int]
    duplicate_rows_removed: int
    invalid_age_fixed: int
    cleaned_shape: tuple[int, int]
    class_counts_cleaned: dict[str, int]
    train_shape: tuple[int, int]
    test_shape: tuple[int, int]
    train_class_counts: dict[int, int]
    test_class_counts: dict[int, int]


def load_raw_dataset() -> pd.DataFrame:
    """Load the UCI Thyroid Disease ('sick') dataset from cache or OpenML."""
    if RAW_DATA_PATH.exists():
        return pd.read_csv(RAW_DATA_PATH)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset = fetch_openml(data_id=OPENML_DATASET_ID, as_frame=True, parser="auto")
    df = dataset.frame.copy()
    df.to_csv(RAW_DATA_PATH, index=False)
    return df


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop all-null/constant columns, fix impossible ages, and remove exact duplicates."""
    raw_shape = df.shape
    missing_counts_raw = {
        col: int(count) for col, count in df.isna().sum().items() if count > 0
    }

    cleaned = df.drop(columns=DROP_COLUMNS, errors="ignore").copy()

    invalid_age_mask = (cleaned["age"] <= 0) | (cleaned["age"] > 120)
    invalid_age_fixed = int(invalid_age_mask.sum())
    cleaned.loc[invalid_age_mask, "age"] = np.nan

    before_dedup = len(cleaned)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    duplicate_rows_removed = before_dedup - len(cleaned)

    info = {
        "raw_shape": raw_shape,
        "missing_counts_raw": missing_counts_raw,
        "invalid_age_fixed": invalid_age_fixed,
        "duplicate_rows_removed": duplicate_rows_removed,
        "cleaned_shape": cleaned.shape,
        "class_counts_cleaned": cleaned[TARGET_COLUMN].value_counts().to_dict(),
    }
    return cleaned, info


def prepare_train_test_data(
    save_test_csv: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, DatasetSummary]:
    """Clean dataset, encode target, perform 80:20 stratified split, and save test_data.csv."""
    raw_df = load_raw_dataset()
    cleaned_df, info = clean_dataset(raw_df)

    X = cleaned_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = cleaned_df[TARGET_COLUMN].map({NEGATIVE_LABEL: 0, POSITIVE_LABEL: 1})
    if y.isna().any():
        raise ValueError("Unexpected target labels encountered while mapping Class.")
    y = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    if save_test_csv:
        test_export = X_test.copy()
        test_export[TARGET_COLUMN] = y_test.map({0: NEGATIVE_LABEL, 1: POSITIVE_LABEL})
        test_export.to_csv(TEST_DATA_PATH, index=False)

    summary = DatasetSummary(
        raw_shape=info["raw_shape"],
        missing_counts_raw=info["missing_counts_raw"],
        duplicate_rows_removed=info["duplicate_rows_removed"],
        invalid_age_fixed=info["invalid_age_fixed"],
        cleaned_shape=info["cleaned_shape"],
        class_counts_cleaned=info["class_counts_cleaned"],
        train_shape=X_train.shape,
        test_shape=X_test.shape,
        train_class_counts=y_train.value_counts().to_dict(),
        test_class_counts=y_test.value_counts().to_dict(),
    )
    return X_train, X_test, y_train, y_test, summary


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """Construct a leak-free ColumnTransformer (fitted strictly on training data inside Pipeline)."""
    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median"))
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="if_binary",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
