from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier

from app.backend.service import PredictionService


def test_prediction_service_loads_promoted_sklearn_package(tmp_path: Path) -> None:
    train = pd.DataFrame(
        {
            "premise": ["a man plays guitar", "the room is empty", "a dog runs"],
            "hypothesis": ["a person makes music", "people are inside", "an animal moves"],
            "label": [0, 2, 0],
        }
    )
    vectorizer = CountVectorizer()
    vectorizer.fit(pd.concat([train["premise"], train["hypothesis"]]))
    features = hstack(
        [
            vectorizer.transform(train["premise"]),
            vectorizer.transform(train["hypothesis"]),
        ],
        format="csr",
    )
    model = DecisionTreeClassifier(random_state=42)
    model.fit(features, train["label"])

    model_dir = tmp_path / "current"
    model_dir.mkdir()
    joblib.dump(model, model_dir / "model.joblib")
    joblib.dump(vectorizer, model_dir / "vectorizer.joblib")

    service = PredictionService(model_dir)
    result = service.predict("A man plays guitar", "A person makes music", "en")

    assert service.is_ready is True
    assert result["label"] in {0, 1, 2}
    assert result["label_name"] in {"entailment", "neutral", "contradiction"}
