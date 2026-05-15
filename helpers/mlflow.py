from __future__ import annotations

import json
import os
import shutil
import subprocess
from tempfile import TemporaryDirectory
from pathlib import Path
from typing import Any

import mlflow
import yaml
from dotenv import load_dotenv

from helpers.metrics import flatten_metrics

load_dotenv()


DEFAULT_TRACKING_URI = "https://dagshub.com/stepan.nazar.23/Contradictory-My-Dear-Watson.mlflow"
DEFAULT_EXPERIMENT_NAME = "contradictory-my-dear-watson"


def setup_mlflow(
    tracking_uri: str = DEFAULT_TRACKING_URI,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    require_credentials: bool = True,
) -> None:
    if require_credentials and tracking_uri.startswith("http"):
        if not os.getenv("MLFLOW_TRACKING_USERNAME") or not os.getenv("MLFLOW_TRACKING_PASSWORD"):
            raise RuntimeError(
                "Set MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD before logging to DagsHub."
            )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def get_git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except Exception:
        return None


def get_dvc_status() -> str | None:
    try:
        completed = subprocess.run(
            ["uv", "run", "dvc", "status"],
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except Exception:
        return None


def log_yaml(path: str | Path, artifact_path: str = "config") -> None:
    mlflow.log_artifact(str(path), artifact_path=artifact_path)


def log_json(payload: dict[str, Any], name: str, artifact_path: str = "metadata") -> Path:
    temp_path = Path(name)
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    try:
        mlflow.log_artifact(str(temp_path), artifact_path=artifact_path)
    finally:
        temp_path.unlink(missing_ok=True)
    return temp_path


def log_model_artifacts(
    artifacts: dict[str, str | Path],
    metadata: dict[str, Any] | None = None,
    artifact_path: str = "model",
) -> None:
    """Log an arbitrary model package to MLflow.

    The caller chooses the files. For example:
    - sklearn text model: model.joblib, vectorizer.joblib, metadata.json
    - TREE-G model: model.joblib, graph_encoder.joblib, parser_config.json
    - transformer: model weights/tokenizer/config files

    FastAPI serving still needs a loader that understands the promoted package.
    """
    with TemporaryDirectory() as temp_dir:
        package_dir = Path(temp_dir) / artifact_path
        package_dir.mkdir(parents=True, exist_ok=True)

        for target_name, source_path in artifacts.items():
            source = Path(source_path)
            if not source.exists():
                raise FileNotFoundError(f"Model artifact does not exist: {source}")
            destination = package_dir / target_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)

        if metadata is not None:
            with (package_dir / "metadata.json").open("w", encoding="utf-8") as file:
                json.dump(metadata, file, indent=2, ensure_ascii=False)

        mlflow.log_artifacts(str(package_dir), artifact_path=artifact_path)


def log_metrics(metrics: dict[str, Any]) -> None:
    mlflow.log_metrics(flatten_metrics(metrics))
    mlflow.log_dict(metrics, "metrics/metrics.json")


def log_common_context(
    config_path: str | Path = "configs/data_split.yaml",
    split_metadata_path: str | Path = "data/processed/split_metadata.json",
) -> None:
    mlflow.set_tag("git_commit", get_git_commit() or "unknown")
    dvc_status = get_dvc_status()
    if dvc_status:
        mlflow.set_tag("dvc_status", dvc_status[:250])

    config_path = Path(config_path)
    if config_path.exists():
        mlflow.log_artifact(str(config_path), artifact_path="config")
        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        mlflow.log_dict(config, "config/data_split.parsed.json")

    split_metadata_path = Path(split_metadata_path)
    if split_metadata_path.exists():
        mlflow.log_artifact(str(split_metadata_path), artifact_path="data")


def start_notebook_run(
    run_name: str,
    tags: dict[str, str] | None = None,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    require_credentials: bool = True,
):
    setup_mlflow(tracking_uri, experiment_name, require_credentials=require_credentials)
    run = mlflow.start_run(run_name=run_name)
    mlflow.set_tag("stage", "notebook_research")
    for key, value in (tags or {}).items():
        mlflow.set_tag(key, value)
    return run
