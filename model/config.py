"""Central configuration for dataset paths, constants, and feature schemas."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"
RESULTS_DIR = PROJECT_ROOT / "results"

RAW_DATA_PATH = DATA_DIR / "thyroid_sick_raw.csv"
TEST_DATA_PATH = PROJECT_ROOT / "test_data.csv"

OPENML_DATASET_ID = 38
RANDOM_STATE = 42
TEST_SIZE = 0.20

TARGET_COLUMN = "Class"
POSITIVE_LABEL = "sick"
NEGATIVE_LABEL = "negative"

DROP_COLUMNS = ["TBG", "TBG_measured"]

NUMERIC_FEATURES = ["age", "TSH", "T3", "TT4", "T4U", "FTI"]

CATEGORICAL_FEATURES = [
    "sex",
    "on_thyroxine",
    "query_on_thyroxine",
    "on_antithyroid_medication",
    "sick",
    "pregnant",
    "thyroid_surgery",
    "I131_treatment",
    "query_hypothyroid",
    "query_hyperthyroid",
    "lithium",
    "goitre",
    "tumor",
    "hypopituitary",
    "psych",
    "TSH_measured",
    "T3_measured",
    "TT4_measured",
    "T4U_measured",
    "FTI_measured",
    "referral_source",
]
