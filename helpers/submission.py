from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_submission(sample_submission: pd.DataFrame, predictions) -> pd.DataFrame:
    if len(sample_submission) != len(predictions):
        raise ValueError(
            f"Expected {len(sample_submission)} predictions, got {len(predictions)}."
        )
    submission = sample_submission[["id"]].copy()
    submission["prediction"] = pd.Series(predictions, index=submission.index).astype(int)
    validate_submission(submission)
    return submission


def validate_submission(submission: pd.DataFrame) -> None:
    expected_columns = ["id", "prediction"]
    if list(submission.columns) != expected_columns:
        raise ValueError(f"Submission columns must be {expected_columns}.")
    if submission["id"].isna().any():
        raise ValueError("Submission contains missing ids.")
    if submission["id"].duplicated().any():
        raise ValueError("Submission contains duplicated ids.")
    invalid_labels = set(submission["prediction"].dropna().astype(int)) - {0, 1, 2}
    if invalid_labels:
        raise ValueError(f"Submission contains invalid labels: {sorted(invalid_labels)}.")


def save_submission(submission: pd.DataFrame, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    validate_submission(submission)
    submission.to_csv(output_path, index=False)
    return output_path
