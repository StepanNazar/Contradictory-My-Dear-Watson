from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

LABELS = [0, 1, 2]


def compute_classification_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_micro": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_micro": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
    }


def compute_confusion_matrix(y_true, y_pred) -> list[list[int]]:
    return confusion_matrix(y_true, y_pred, labels=LABELS).astype(int).tolist()


def compute_per_language_metrics(
    validation_df: pd.DataFrame,
    y_true,
    y_pred,
    language_column: str = "lang_abv",
) -> dict[str, dict[str, Any]]:
    if language_column not in validation_df.columns:
        return {}

    frame = validation_df[[language_column]].copy()
    frame["y_true"] = list(y_true)
    frame["y_pred"] = list(y_pred)

    result: dict[str, dict[str, Any]] = {}
    for language, group in frame.groupby(language_column):
        result[str(language)] = {
            **compute_classification_metrics(group["y_true"], group["y_pred"]),
            "confusion_matrix": compute_confusion_matrix(group["y_true"], group["y_pred"]),
            "support": int(len(group)),
        }
    return result


def compute_all_metrics(validation_df: pd.DataFrame, y_pred, label_column: str = "label") -> dict[str, Any]:
    y_true = validation_df[label_column].astype(int)
    return {
        **compute_classification_metrics(y_true, y_pred),
        "confusion_matrix": compute_confusion_matrix(y_true, y_pred),
        "per_language": compute_per_language_metrics(validation_df, y_true, y_pred),
    }


def flatten_metrics(metrics: dict[str, Any], prefix: str = "") -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in metrics.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_metrics(value, name))
        elif isinstance(value, (int, float)) and key != "support":
            flat[name] = float(value)
    return flat
