# Contradictory My Dear Watson

Notebook-first Kaggle experimentation workflow for
[Contradictory, My Dear Watson](https://www.kaggle.com/competitions/contradictory-my-dear-watson).

## What This Project Does

```text
DVC       -> raw Kaggle data and one shared train/validation/test split
notebooks -> research experiments and Kaggle code submissions
MLflow    -> params, metrics, model artifacts, submissions
FastAPI   -> serves one promoted MLflow model from models/current
```

The project intentionally avoids a heavy production training framework. Experiments live in notebooks and MLflow; DVC is only for data/split reproducibility.

## Structure

```text
configs/data_split.yaml  data paths and split settings
helpers/                 reusable notebook helpers
scripts/                 data preparation and model promotion scripts
notebooks/               research experiments
app/backend/             FastAPI service
models/current/          promoted local model copy for FastAPI
docs/onboarding.md       full setup guide
```

## Quickstart

```bash
uv sync
uv run nbstripout --install
```

Create `.env` locally:

```env
MLFLOW_TRACKING_USERNAME=<your_dagshub_username>
MLFLOW_TRACKING_PASSWORD=<your_dagshub_token>
```

Configure DVC credentials locally:

```bash
uv run dvc remote modify origin --local access_key_id <your_dagshub_token>
uv run dvc remote modify origin --local secret_access_key <your_dagshub_token>
```

Then pull data if it already exists:

```bash
uv run dvc pull -r origin
```

Or, if adding data for the first time, place Kaggle CSVs in `data/raw/` and run:

```bash
uv run dvc add data/raw
uv run dvc repro
uv run dvc push -r origin
```

## Experiment Flow

Run:

```text
notebooks/01_bow_decision_tree_research.ipynb
```

The notebook runs several baseline experiments and logs metrics, model artifacts, metadata, and local prediction CSVs to DagsHub MLflow.

Kaggle final submission is done through a Kaggle notebook/code submission, not by manually uploading the local CSV. Copy the selected experiment and required helper functions into the Kaggle notebook, produce `submission.csv` there, and submit the notebook output.

## Serve A Promoted Model

Pick a run in MLflow and promote it:

```bash
uv run python scripts/promote_model.py --run-id <mlflow_run_id>
```

Start API:

```bash
docker compose up --build api
```

FastAPI currently serves the baseline `sklearn_text_pair_vectorizer` package. Other model packages can be logged/promoted, but need a matching loader in `app/backend/service.py`.

## Tests

```bash
uv run pytest
```

## Full Setup

See [docs/onboarding.md](docs/onboarding.md) for account setup, DagsHub/DVC details, Kaggle notebook submission instructions, and team workflow.
