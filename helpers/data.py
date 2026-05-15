from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_processed_data(processed_dir: str | Path = "data/processed") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load DVC-versioned train/validation/test splits."""
    base = Path(processed_dir)
    return (
        pd.read_csv(base / "train_split.csv"),
        pd.read_csv(base / "val_split.csv"),
        pd.read_csv(base / "kaggle_test.csv"),
    )


def load_sample_submission(path: str | Path = "data/raw/sample_submission.csv") -> pd.DataFrame:
    return pd.read_csv(path)
