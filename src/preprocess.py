"""Load, clean, and vectorize the SMS spam dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "spam.csv"

# The Kaggle file ships two useful columns plus three empty "Unnamed" columns.
LABEL_COLUMN = "v1"
MESSAGE_COLUMN = "v2"
LABEL_MAP = {"ham": 0, "spam": 1}


def load_clean_data(
    data_path: Path = DEFAULT_DATA_PATH,
) -> tuple[pd.DataFrame, int]:
    """Load the raw CSV, keep the two real columns, and drop duplicates.

    Returns the cleaned frame with a numeric ``label`` (0 = ham, 1 = spam)
    and a ``message`` text column, plus the number of duplicate rows removed.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find the dataset at {data_path}")

    # The file is not UTF-8; latin-1 reads every byte without errors.
    raw = pd.read_csv(data_path, encoding="latin-1")
    missing = {LABEL_COLUMN, MESSAGE_COLUMN} - set(raw.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    data = raw[[LABEL_COLUMN, MESSAGE_COLUMN]].copy()
    data.columns = ["label", "message"]
    data["label"] = data["label"].str.strip().str.lower().map(LABEL_MAP)
    data = data.dropna(subset=["label", "message"])
    data["label"] = data["label"].astype(int)

    duplicate_count = int(data.duplicated().sum())
    clean_data = data.drop_duplicates().reset_index(drop=True)
    return clean_data, duplicate_count


def split_features_target(
    data: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Return the raw message text X and binary spam target y."""
    X = data["message"].copy()
    y = data["label"].copy()
    return X, y


def build_vectorizer() -> TfidfVectorizer:
    """Turn message text into TF-IDF features.

    Lower-casing, English stop-word removal, and unigram+bigram terms give
    a naive-Bayes classifier enough signal without exploding the vocabulary.
    """
    return TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )
