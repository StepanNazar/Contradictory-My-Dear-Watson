from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    train = pd.read_csv(config["raw"]["train_path"])
    test = pd.read_csv(config["raw"]["test_path"])
    split = config["split"]

    train_split, val_split = train_test_split(
        train,
        test_size=split["test_size"],
        random_state=split["random_state"],
        stratify=train[split["stratify_by"]],
    )

    processed = config["processed"]
    for key in ["train_split_path", "val_split_path", "kaggle_test_path", "split_metadata_path"]:
        Path(processed[key]).parent.mkdir(parents=True, exist_ok=True)

    train_split.to_csv(processed["train_split_path"], index=False)
    val_split.to_csv(processed["val_split_path"], index=False)
    test.to_csv(processed["kaggle_test_path"], index=False)

    metadata = {
        "strategy": split["strategy"],
        "stratify_by": split["stratify_by"],
        "test_size": split["test_size"],
        "random_state": split["random_state"],
        "train_rows": int(len(train_split)),
        "val_rows": int(len(val_split)),
        "kaggle_test_rows": int(len(test)),
    }
    with Path(processed["split_metadata_path"]).open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
