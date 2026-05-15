from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

TRAIN_COLUMNS = {"id", "premise", "hypothesis", "language", "lang_abv", "label"}
TEST_COLUMNS = {"id", "premise", "hypothesis", "language", "lang_abv"}
VALID_LABELS = {0, 1, 2}


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def validate(config: dict) -> dict:
    train = pd.read_csv(config["raw"]["train_path"])
    test = pd.read_csv(config["raw"]["test_path"])
    sample_submission = pd.read_csv(config["raw"]["sample_submission_path"])

    errors: list[str] = []

    missing_train = TRAIN_COLUMNS - set(train.columns)
    if missing_train:
        errors.append(f"train.csv missing columns: {sorted(missing_train)}")

    missing_test = TEST_COLUMNS - set(test.columns)
    if missing_test:
        errors.append(f"test.csv missing columns: {sorted(missing_test)}")

    if {"id", "prediction"} - set(sample_submission.columns):
        errors.append("sample_submission.csv must contain id and prediction columns")

    for name, frame in [("train.csv", train), ("test.csv", test)]:
        if "id" in frame.columns and frame["id"].duplicated().any():
            errors.append(f"{name} contains duplicated ids")
        for column in ["premise", "hypothesis"]:
            if column in frame.columns and frame[column].isna().any():
                errors.append(f"{name} contains missing values in {column}")

    if "label" in train.columns:
        invalid_labels = set(train["label"].dropna().astype(int)) - VALID_LABELS
        if invalid_labels:
            errors.append(f"train.csv contains invalid labels: {sorted(invalid_labels)}")

    return {
        "ok": not errors,
        "errors": errors,
        "rows": {
            "train": int(len(train)),
            "test": int(len(test)),
            "sample_submission": int(len(sample_submission)),
        },
        "languages": sorted(train["lang_abv"].dropna().unique().tolist()) if "lang_abv" in train else [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    report = validate(config)
    output_path = Path(config["processed"]["validation_report_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    if not report["ok"]:
        raise SystemExit(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
