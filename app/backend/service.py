from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from scipy.sparse import hstack


LABEL_MAPPING = {
    0: "entailment",
    1: "neutral",
    2: "contradiction",
}


class PredictionService:
    def __init__(self, model_dir: str | Path = "models/current") -> None:
        self.model_dir = Path(model_dir)
        self.model = None
        self.vectorizer = None
        self.metadata: dict[str, Any] = {}
        self.load_error: str | None = None
        self.reload()

    def reload(self) -> None:
        try:
            model_path = self.model_dir / "model.joblib"
            vectorizer_path = self.model_dir / "vectorizer.joblib"
            if not model_path.exists() or not vectorizer_path.exists():
                raise FileNotFoundError(
                    f"Expected {model_path} and {vectorizer_path}. Run scripts/promote_model.py first."
                )
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            metadata_path = self.model_dir / "metadata.json"
            if metadata_path.exists():
                with metadata_path.open("r", encoding="utf-8") as file:
                    self.metadata = json.load(file)
            else:
                self.metadata = {}
            serving_type = self.metadata.get("serving_type", "sklearn_text_pair_vectorizer")
            if serving_type != "sklearn_text_pair_vectorizer":
                raise ValueError(
                    f"Unsupported serving_type={serving_type!r}. "
                    "Current FastAPI loader supports only 'sklearn_text_pair_vectorizer'."
                )
            self.load_error = None
        except Exception as exc:
            self.model = None
            self.vectorizer = None
            self.metadata = {}
            self.load_error = str(exc)

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.vectorizer is not None

    def model_info(self) -> dict[str, Any]:
        if not self.is_ready:
            return {"loaded": False, "error": self.load_error}
        return {
            "loaded": True,
            "model_dir": str(self.model_dir),
            "metadata": self.metadata,
            "labels": LABEL_MAPPING,
        }

    @staticmethod
    def normalize_text(value: str) -> str:
        return " ".join(str(value).lower().split())

    def _features(self, premise: str, hypothesis: str):
        if self.vectorizer is None:
            raise RuntimeError(self.load_error or "Vectorizer is not loaded.")
        frame = pd.DataFrame(
            {
                "premise": [self.normalize_text(premise)],
                "hypothesis": [self.normalize_text(hypothesis)],
            }
        )
        return hstack(
            [
                self.vectorizer.transform(frame["premise"]),
                self.vectorizer.transform(frame["hypothesis"]),
            ],
            format="csr",
        )

    def predict(self, premise: str, hypothesis: str, language: str | None = None) -> dict[str, Any]:
        if not self.is_ready:
            raise RuntimeError(self.load_error or "Model is not loaded.")

        features = self._features(premise, hypothesis)
        prediction = int(self.model.predict(features)[0])
        probabilities = None
        if hasattr(self.model, "predict_proba"):
            raw_probabilities = self.model.predict_proba(features)[0]
            classes = [int(item) for item in self.model.classes_]
            probabilities = {
                LABEL_MAPPING[label]: float(raw_probabilities[index])
                for index, label in enumerate(classes)
            }

        return {
            "label": prediction,
            "label_name": LABEL_MAPPING[prediction],
            "probabilities": probabilities,
        }
