import pandas as pd
import pytest

from helpers.submission import build_submission, validate_submission


def test_build_submission_uses_sample_ids_and_predictions() -> None:
    sample = pd.DataFrame({"id": [10, 11], "prediction": [0, 0]})

    submission = build_submission(sample, [1, 2])

    assert list(submission.columns) == ["id", "prediction"]
    assert submission.to_dict(orient="records") == [
        {"id": 10, "prediction": 1},
        {"id": 11, "prediction": 2},
    ]


def test_validate_submission_rejects_invalid_labels() -> None:
    submission = pd.DataFrame({"id": [1], "prediction": [9]})

    with pytest.raises(ValueError, match="invalid labels"):
        validate_submission(submission)
