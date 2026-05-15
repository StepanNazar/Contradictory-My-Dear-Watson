import pandas as pd

from helpers.metrics import compute_all_metrics, flatten_metrics


def test_compute_all_metrics_includes_global_and_per_language_values() -> None:
    validation = pd.DataFrame(
        {
            "label": [0, 1, 2, 0],
            "lang_abv": ["en", "en", "fr", "fr"],
        }
    )
    predictions = [0, 2, 2, 0]

    metrics = compute_all_metrics(validation, predictions)
    flat = flatten_metrics(metrics)

    assert metrics["accuracy"] == 0.75
    assert "accuracy_macro" not in metrics
    assert "accuracy_micro" not in metrics
    assert "precision_micro" in metrics
    assert "recall_micro" in metrics
    assert "f1_micro" in metrics
    assert metrics["confusion_matrix"] == [[2, 0, 0], [0, 0, 1], [0, 0, 1]]
    assert "confusion_matrix" in metrics["per_language"]["en"]
    assert metrics["per_language"]["en"]["support"] == 2
    assert "accuracy_macro" not in metrics["per_language"]["en"]
    assert "accuracy_micro" not in metrics["per_language"]["en"]
    assert "per_language.en.accuracy" in flat
    assert "per_language.en.accuracy_macro" not in flat
    assert "per_language.en.accuracy_micro" not in flat
    assert "per_language.en.f1_micro" in flat
