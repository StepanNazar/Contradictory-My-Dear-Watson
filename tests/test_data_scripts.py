from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import yaml

from scripts.prepare_splits import main as prepare_main
from scripts.validate_data import main as validate_main


def write_config(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "data" / "raw"
    processed_dir = tmp_path / "data" / "processed"
    raw_dir.mkdir(parents=True)
    shutil.copy("tests/fixtures/train.csv", raw_dir / "train.csv")
    shutil.copy("tests/fixtures/test.csv", raw_dir / "test.csv")
    shutil.copy("tests/fixtures/sample_submission.csv", raw_dir / "sample_submission.csv")

    config = {
        "raw": {
            "train_path": str(raw_dir / "train.csv"),
            "test_path": str(raw_dir / "test.csv"),
            "sample_submission_path": str(raw_dir / "sample_submission.csv"),
        },
        "processed": {
            "train_split_path": str(processed_dir / "train_split.csv"),
            "val_split_path": str(processed_dir / "val_split.csv"),
            "kaggle_test_path": str(processed_dir / "kaggle_test.csv"),
            "validation_report_path": str(processed_dir / "data_validation.json"),
            "split_metadata_path": str(processed_dir / "split_metadata.json"),
        },
        "split": {
            "strategy": "stratified_split",
            "stratify_by": "label",
            "test_size": 0.25,
            "random_state": 42,
        },
    }
    path = tmp_path / "config.yaml"
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file)
    return path


def test_validate_and_prepare_scripts_create_processed_split_files(tmp_path: Path, monkeypatch) -> None:
    config_path = write_config(tmp_path)

    monkeypatch.setattr("sys.argv", ["validate_data.py", "--config", str(config_path)])
    validate_main()

    monkeypatch.setattr("sys.argv", ["prepare_splits.py", "--config", str(config_path)])
    prepare_main()

    processed = tmp_path / "data" / "processed"
    assert (processed / "data_validation.json").exists()
    assert (processed / "train_split.csv").exists()
    assert (processed / "val_split.csv").exists()
    assert (processed / "kaggle_test.csv").exists()

    with (processed / "split_metadata.json").open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    assert metadata["random_state"] == 42
    assert len(pd.read_csv(processed / "kaggle_test.csv")) == 2
