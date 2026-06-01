from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import nltk
from sentence_transformers import SentenceTransformer


LABEL_MAPPING = {
    0: "entailment",
    1: "neutral",
    2: "contradiction",
}

E5_MODEL_NAME = "intfloat/multilingual-e5-small"


class PredictionService:
    def __init__(self, model_dir: str | Path = "models/current") -> None:
        self.model_dir = Path(model_dir)
        self.base_models = {}
        self.meta_model = None
        self.label_encoder = None
        self.st_model = None
        self.metadata: dict[str, Any] = {}
        self.load_error: str | None = None
        
        # Download NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt')
            nltk.download('punkt_tab')
            
        self.reload()

    def reload(self) -> None:
        try:
            # Check for stacked model files
            required_files = [
                "base_xgboost.joblib", "base_lightgbm.joblib", "base_svm.joblib", 
                "meta_model.joblib", "label_encoder.joblib"
            ]
            for f in required_files:
                if not (self.model_dir / f).exists():
                    raise FileNotFoundError(f"Missing required model file: {f}")
            
            # Load models
            self.base_models = {
                "xgboost": joblib.load(self.model_dir / "base_xgboost.joblib"),
                "lightgbm": joblib.load(self.model_dir / "base_lightgbm.joblib"),
                "svm": joblib.load(self.model_dir / "base_svm.joblib"),
            }
            self.meta_model = joblib.load(self.model_dir / "meta_model.joblib")
            self.label_encoder = joblib.load(self.model_dir / "label_encoder.joblib")
            
            # Load metadata
            metadata_path = self.model_dir / "metadata.json"
            if metadata_path.exists():
                with metadata_path.open("r", encoding="utf-8") as file:
                    self.metadata = json.load(file)
            else:
                self.metadata = {"serving_type": "stacked_e5_model"}
            
            # Load Sentence Transformer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.st_model = SentenceTransformer(E5_MODEL_NAME, device=device)
            
            self.load_error = None
        except Exception as exc:
            self.base_models = {}
            self.meta_model = None
            self.label_encoder = None
            self.st_model = None
            self.metadata = {}
            self.load_error = str(exc)

    @property
    def is_ready(self) -> bool:
        return self.meta_model is not None and self.st_model is not None

    def model_info(self) -> dict[str, Any]:
        if not self.is_ready:
            return {"loaded": False, "error": self.load_error}
        return {
            "loaded": True,
            "model_dir": str(self.model_dir),
            "metadata": self.metadata,
            "labels": LABEL_MAPPING,
        }

    def _get_embeddings(self, texts: list[str]) -> np.ndarray:
        prefixed = [f"query: {t}" for t in texts]
        return self.st_model.encode(prefixed, normalize_embeddings=True)

    def _features(self, p_emb: np.ndarray, h_emb: np.ndarray) -> np.ndarray:
        return np.hstack([
            p_emb, 
            h_emb, 
            np.abs(p_emb - h_emb), 
            p_emb * h_emb
        ])

    def predict(self, premise: str, hypothesis: str, language: str | None = None) -> dict[str, Any]:
        if not self.is_ready:
            raise RuntimeError(self.load_error or "Model is not loaded.")

        # 1. Embeddings
        embs = self._get_embeddings([premise, hypothesis])
        p_emb, h_emb = embs[0:1], embs[1:2]
        
        # 2. Features
        x_base = self._features(p_emb, h_emb)
        
        # 3. Base Model Probs
        base_probs = []
        for name in ["xgboost", "lightgbm", "svm"]:
            base_probs.append(self.base_models[name].predict_proba(x_base))
            
        # 4. Language Feature
        lang_code = "en" if not language else language
        try:
            lang_idx = self.label_encoder.transform([lang_code])[0]
        except ValueError:
            # Fallback to English if language unknown
            lang_idx = self.label_encoder.transform(["en"])[0]
        lang_feat = np.array([[lang_idx]])
        
        # 5. Meta Features
        x_meta = np.hstack(base_probs + [lang_feat])
        
        # 6. Final Prediction
        prediction = int(self.meta_model.predict(x_meta)[0])
        raw_probabilities = self.meta_model.predict_proba(x_meta)[0]
        
        probabilities = {
            LABEL_MAPPING[i]: float(raw_probabilities[i])
            for i in range(len(LABEL_MAPPING))
        }

        return {
            "label": prediction,
            "label_name": LABEL_MAPPING[prediction],
            "probabilities": probabilities,
        }

    def detect_contradictions(self, text: str, language: str) -> dict[str, Any]:
        if not self.is_ready:
            raise RuntimeError(self.load_error or "Model is not loaded.")
            
        sentences = nltk.sent_tokenize(text)
        if len(sentences) < 2:
            return {"contradictions": [], "total_pairs_checked": 0}
            
        # Generate pairs (order matters for NLI, but we'll check both ways or just all combinations)
        import itertools
        pairs = list(itertools.combinations(sentences, 2))
        
        results = []
        # Optimization: batch embedding
        all_sentences = list(set(sentences))
        sent_to_emb = {s: e for s, e in zip(all_sentences, self._get_embeddings(all_sentences))}
        
        try:
            lang_idx = self.label_encoder.transform([language])[0]
        except ValueError:
            lang_idx = self.label_encoder.transform(["en"])[0]
        lang_feat_val = np.array([[lang_idx]])

        for s1, s2 in pairs:
            # We check both directions: s1->s2 and s2->s1
            for p, h in [(s1, s2), (s2, s1)]:
                p_emb = sent_to_emb[p].reshape(1, -1)
                h_emb = sent_to_emb[h].reshape(1, -1)
                
                x_base = self._features(p_emb, h_emb)
                
                base_probs = []
                for name in ["xgboost", "lightgbm", "svm"]:
                    base_probs.append(self.base_models[name].predict_proba(x_base))
                
                x_meta = np.hstack(base_probs + [lang_feat_val])
                
                prob_contradiction = float(self.meta_model.predict_proba(x_meta)[0][2]) # Index 2 is contradiction
                
                if prob_contradiction > 0.5:
                    results.append({
                        "premise": p,
                        "hypothesis": h,
                        "probability": prob_contradiction
                    })
                    
        # Sort by probability descending
        results.sort(key=lambda x: x["probability"], reverse=True)
        
        return {
            "contradictions": results,
            "total_pairs_checked": len(pairs) * 2
        }
