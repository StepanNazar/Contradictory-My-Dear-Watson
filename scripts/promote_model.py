from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import mlflow
from dotenv import load_dotenv

DEFAULT_TRACKING_URI = "https://dagshub.com/stepan.nazar.23/Contradictory-My-Dear-Watson.mlflow"

load_dotenv()


def require_mlflow_credentials(tracking_uri: str) -> None:
    if tracking_uri.startswith("http"):
        if not os.getenv("MLFLOW_TRACKING_USERNAME") or not os.getenv("MLFLOW_TRACKING_PASSWORD"):
            raise RuntimeError(
                "Set MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD before downloading from DagsHub MLflow."
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True, help="MLflow run id to promote.")
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--artifact-path", default="model", help="MLflow artifact folder to promote.")
    parser.add_argument("--target-dir", default="models/current")
    args = parser.parse_args()

    require_mlflow_credentials(args.tracking_uri)
    mlflow.set_tracking_uri(args.tracking_uri)

    target_dir = Path(args.target_dir)
    download_root = target_dir.parent / ".downloaded"
    if download_root.exists():
        shutil.rmtree(download_root)
    download_root.mkdir(parents=True, exist_ok=True)

    downloaded_path = Path(
        mlflow.artifacts.download_artifacts(
            run_id=args.run_id,
            artifact_path=args.artifact_path,
            dst_path=str(download_root),
        )
    )

    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(downloaded_path, target_dir)

    metadata_path = target_dir / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
    metadata.update(
        {
            "mlflow_run_id": args.run_id,
            "mlflow_tracking_uri": args.tracking_uri,
            "mlflow_artifact_path": args.artifact_path,
        }
    )
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    shutil.rmtree(download_root)
    print(f"Promoted MLflow run {args.run_id} to {target_dir}")


if __name__ == "__main__":
    main()
