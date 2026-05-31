import argparse
import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sentence_transformers import SentenceTransformer

from helpers.data import load_processed_data
from helpers.metrics import compute_all_metrics
from helpers.mlflow import log_common_context, log_metrics, log_model_artifacts, setup_mlflow

# Constants
MODEL_NAME = "intfloat/multilingual-e5-small"
EXPECTED_CLASSES = np.array([0, 1, 2])
MODELS_DIR = Path("models/current")

# Fixed Hyperparameters
PARAMS_SVM = {'C': 1.508238815114371, 'kernel': 'linear', 'probability': True, 'random_state': 42}
PARAMS_XGB = {
    'n_estimators': 372, 'max_depth': 3, 'learning_rate': 0.03762729293261653, 
    'subsample': 0.5910343217723766, 'colsample_bytree': 0.8966157708016231, 
    'random_state': 42, 'eval_metric': 'mlogloss', 'n_jobs': -1
}
PARAMS_LGBM = {
    'n_estimators': 474, 'num_leaves': 126, 'learning_rate': 0.11486230805290915, 
    'random_state': 42, 'verbosity': -1, 'n_jobs': -1
}
PARAMS_META = {
    'n_estimators': 53, 'max_depth': 4, 'learning_rate': 0.06325998591956158, 
    'subsample': 0.9182833675134991, 'random_state': 42, 'eval_metric': 'mlogloss', 'n_jobs': -1
}

def generate_embeddings(texts: list[str], model: SentenceTransformer, prefix: str = "query: ") -> np.ndarray:
    prefixed_texts = [f"{prefix}{text}" for text in texts]
    embeddings = model.encode(prefixed_texts, normalize_embeddings=True, show_progress_bar=True)
    return embeddings

def prepare_features(p, h):
    return np.hstack([
        p, 
        h, 
        np.abs(p - h), 
        p * h
    ])

def train_base_model(model_name, params, x, y):
    if model_name == 'xgboost':
        model = XGBClassifier(**params)
    elif model_name == 'lightgbm':
        model = LGBMClassifier(**params)
    else:
        model = SVC(**params)
    model.fit(x, y)
    return model

def get_oof_predictions(model_name, params, x, y, x_v, x_t):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros((len(x), 3))
    val_probs = np.zeros((len(x_v), 3))
    test_probs = np.zeros((len(x_t), 3))
    
    for train_idx, holdout_idx in skf.split(x, y):
        m = train_base_model(model_name, params, x[train_idx], y.iloc[train_idx])
        
        if not np.array_equal(m.classes_, EXPECTED_CLASSES):
            raise ValueError(f"Model {model_name} has unexpected class order: {m.classes_}")
            
        oof_probs[holdout_idx] = m.predict_proba(x[holdout_idx])
        val_probs += m.predict_proba(x_v) / 5
        test_probs += m.predict_proba(x_t) / 5
        
    return oof_probs, val_probs, test_probs

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    print("Loading data...")
    train_df, val_df, test_df = load_processed_data()
    
    # 2. Generate Embeddings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} on {device}...")
    st_model = SentenceTransformer(MODEL_NAME, device=device)
    
    print("Generating embeddings...")
    train_p = generate_embeddings(train_df['premise'].fillna("").tolist(), st_model)
    train_h = generate_embeddings(train_df['hypothesis'].fillna("").tolist(), st_model)
    val_p = generate_embeddings(val_df['premise'].fillna("").tolist(), st_model)
    val_h = generate_embeddings(val_df['hypothesis'].fillna("").tolist(), st_model)
    test_p = generate_embeddings(test_df['premise'].fillna("").tolist(), st_model)
    test_h = generate_embeddings(test_df['hypothesis'].fillna("").tolist(), st_model)
    
    x_train = prepare_features(train_p, train_h)
    x_val = prepare_features(val_p, val_h)
    x_test = prepare_features(test_p, test_h)
    
    y_train = train_df['label'].astype(int)
    y_val = val_df['label'].astype(int)
    
    # 3. Language Encoding
    print("Encoding languages...")
    le = LabelEncoder()
    train_lang = le.fit_transform(train_df['lang_abv']).reshape(-1, 1)
    val_lang = le.transform(val_df['lang_abv']).reshape(-1, 1)
    test_lang = le.transform(test_df['lang_abv']).reshape(-1, 1)
    joblib.dump(le, MODELS_DIR / 'label_encoder.joblib')
    
    # 4. Define Base Params Mapping
    base_params = {
        'xgboost': PARAMS_XGB,
        'lightgbm': PARAMS_LGBM,
        'svm': PARAMS_SVM
    }
    
    # 5. Generate OOF Predictions
    print("Generating OOF predictions for stacking...")
    meta_train_list = []
    meta_val_list = []
    meta_test_list = []
    
    for m_name in ['xgboost', 'lightgbm', 'svm']:
        print(f"Processing base model: {m_name}")
        o, v, t = get_oof_predictions(m_name, base_params[m_name], x_train, y_train, x_val, x_test)
        meta_train_list.append(o)
        meta_val_list.append(v)
        meta_test_list.append(t)
        
    x_meta_train = np.hstack(meta_train_list + [train_lang])
    x_meta_val = np.hstack(meta_val_list + [val_lang])
    x_meta_test = np.hstack(meta_test_list + [test_lang])
    
    # 6. Final Training on Combined Data (Train + Val)
    print("Final training on combined data (Train + Val)...")
    
    # Combine features and labels
    x_meta_full = np.vstack([x_meta_train, x_meta_val])
    y_full = pd.concat([y_train, y_val], ignore_index=True)
    
    x_train_full = np.vstack([x_train, x_val])
    
    # Train meta-model on all available meta-features
    meta_model = XGBClassifier(**PARAMS_META)
    meta_model.fit(x_meta_full, y_full)
    joblib.dump(meta_model, MODELS_DIR / "meta_model.joblib")
    
    # Train base models on all available raw features
    base_paths = {}
    for m_name in ['xgboost', 'lightgbm', 'svm']:
        print(f"Final training base model: {m_name}")
        m = train_base_model(m_name, base_params[m_name], x_train_full, y_full)
        p = MODELS_DIR / f"base_{m_name}.joblib"
        joblib.dump(m, p)
        base_paths[f"base_{m_name}.joblib"] = p
        
    # 7. Evaluation (Note: this is on the meta-model trained on x_meta_full, so it's biased but kept for script flow)
    # The 'metrics' reported will be from the meta-model's performance on the data it was trained on.
    val_preds = meta_model.predict(x_meta_val)
    metrics = compute_all_metrics(val_df, val_preds)
    print(f"Training-set Accuracy (on Val portion): {metrics['accuracy']:.4f}")
    
    # 8. Logging to MLflow
    setup_mlflow()
    with mlflow.start_run(run_name="final_stacked_model"):
        mlflow.log_params({"meta_" + k: v for k, v in PARAMS_META.items()})
        log_metrics(metrics)
        log_common_context()
        
        artifacts = {
            "meta_model.joblib": MODELS_DIR / "meta_model.joblib",
            "label_encoder.joblib": MODELS_DIR / "label_encoder.joblib",
            "base_xgboost.joblib": MODELS_DIR / "base_xgboost.joblib",
            "base_lightgbm.joblib": MODELS_DIR / "base_lightgbm.joblib",
            "base_svm.joblib": MODELS_DIR / "base_svm.joblib",
        }
        
        log_model_artifacts(
            artifacts=artifacts,
            metadata={
                "serving_type": "stacked_e5_model",
                "base_models": list(base_params.keys()),
                "e5_model": MODEL_NAME
            }
        )

    print(f"Training completed. Models saved to {MODELS_DIR}")

if __name__ == "__main__":
    main()
