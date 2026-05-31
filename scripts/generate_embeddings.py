import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sentence_transformers import SentenceTransformer


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def generate_embeddings(texts: list[str], model: SentenceTransformer, prefix: str = "query: ") -> np.ndarray:
    # E5 models require a prefix
    prefixed_texts = [f"{prefix}{text}" for text in texts]
    embeddings = model.encode(prefixed_texts, normalize_embeddings=True, show_progress_bar=True)
    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    processed_paths = config["processed"]

    # Define output directory
    output_dir = Path("embeddings/e5")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    model_name = "intfloat/multilingual-e5-small"
    model = SentenceTransformer(model_name, device=device)

    datasets = {
        "train": processed_paths["train_split_path"],
        "val": processed_paths["val_split_path"],
        "test": processed_paths["kaggle_test_path"],
    }

    for name, path in datasets.items():
        print(f"Processing {name} dataset from {path}...")
        df = pd.read_csv(path)
        
        # Fill NaN values if any
        df["premise"] = df["premise"].fillna("")
        df["hypothesis"] = df["hypothesis"].fillna("")

        print(f"Generating premise embeddings for {name}...")
        premise_embeddings = generate_embeddings(df["premise"].tolist(), model)
        np.save(output_dir / f"{name}_premise.npy", premise_embeddings)

        print(f"Generating hypothesis embeddings for {name}...")
        hypothesis_embeddings = generate_embeddings(df["hypothesis"].tolist(), model)
        np.save(output_dir / f"{name}_hypothesis.npy", hypothesis_embeddings)

    print("Embeddings generation completed.")


if __name__ == "__main__":
    main()
