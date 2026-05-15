# Onboarding

This guide is for a new teammate joining the project from a clean machine.

## Project Rules

DVC owns data and shared splits. MLflow owns experiment results and model artifacts. Notebooks own research. FastAPI serves only the model explicitly promoted from MLflow into `models/current`.

Do not commit secrets, downloaded Kaggle CSV files, generated processed data, local MLflow folders, submissions, or promoted model files.

## 1. Accounts And Access

Create or confirm these accounts:

```text
GitHub   -> source code repository
DagsHub  -> DVC remote storage and MLflow tracking
Kaggle   -> competition data and notebook/code submissions
```

Required access:

```text
GitHub repo:
  https://github.com/StepanNazar/Contradictory-My-Dear-Watson

DagsHub repo:
  stepan.nazar.23/Contradictory-My-Dear-Watson

Kaggle competition:
  https://www.kaggle.com/competitions/contradictory-my-dear-watson
```

Ask the project owner to add you:

```text
GitHub:
  as collaborator or via the team that has write access to the repository

DagsHub:
  as collaborator with write access to the matching DagsHub repository

Kaggle:
  join the competition and, if the team uses a shared Kaggle team, ask for an invite to that Kaggle team
```

Each person uses their own DagsHub token. Do not share tokens.

## 2. Clone And Install

Clone the GitHub repository:

```bash
git clone https://github.com/StepanNazar/Contradictory-My-Dear-Watson.git
cd Contradictory-My-Dear-Watson
```

Install dependencies:

```bash
uv sync
```

If you will edit notebooks, install the notebook output cleaner once locally:

```bash
uv run nbstripout --install
uv run nbstripout --status
```

`nbstripout --install` modifies your local `.git/config`, so every teammate should run it once.

## 3. DagsHub Token And `.env`

Create a DagsHub access token in your DagsHub account settings.

Create a local `.env` file in the repository root:

```env
MLFLOW_TRACKING_USERNAME=<your_dagshub_username>
MLFLOW_TRACKING_PASSWORD=<your_dagshub_token>
```

`.env` is ignored by Git. Never commit it.

The code uses `python-dotenv`, so notebooks and `scripts/promote_model.py` read `.env` automatically.

## 4. Configure DVC Remote Credentials

The shared DVC remote is already configured in `.dvc/config` without secrets. You only need local credentials.

For DagsHub S3/DVC remote, use your DagsHub token as both access key and secret key:

```bash
uv run dvc remote modify origin --local access_key_id <your_dagshub_token>
uv run dvc remote modify origin --local secret_access_key <your_dagshub_token>
```

This writes to `.dvc/config.local`, which is ignored by Git.

Check the remote:

```bash
uv run dvc remote list
uv run dvc remote default
```

## 5. Get Data

Preferred path if another teammate already pushed data to DVC:

```bash
uv run dvc pull -r origin
```

After this, you should have:

```text
data/raw/train.csv
data/raw/test.csv
data/raw/sample_submission.csv
data/processed/train_split.csv
data/processed/val_split.csv
data/processed/kaggle_test.csv
data/processed/split_metadata.json
```

If data has not been pushed to DVC yet, download it manually from Kaggle and place it here:

```text
data/raw/train.csv
data/raw/test.csv
data/raw/sample_submission.csv
```

Then the first teammate with local data runs:

```bash
uv run dvc add data/raw
uv run dvc repro
uv run dvc push -r origin
```

Commit DVC metadata, not CSV files:

```bash
git add data/raw.dvc dvc.yaml dvc.lock .gitignore configs/data_split.yaml
git commit -m "Track Kaggle data with DVC"
git push
```

## 6. Recreate Shared Splits

If raw data or `configs/data_split.yaml` changes, regenerate DVC outputs:

```bash
uv run dvc repro
uv run dvc push -r origin
```

Other teammates then pull the updated data:

```bash
git pull
uv run dvc pull -r origin
```

## 7. Run Notebook Experiments

Open:

```text
notebooks/01_bow_decision_tree_research.ipynb
```

Run all cells. The notebook:

```text
loads DVC processed splits
runs several baseline experiments
computes global and per-language metrics
creates local prediction CSVs for MLflow/debugging
logs params, metrics, model artifacts, metadata, and submissions to DagsHub MLflow
```

MLflow tracking URL:

```text
https://dagshub.com/stepan.nazar.23/Contradictory-My-Dear-Watson.mlflow
```

## 8. Kaggle Submissions

This competition expects a **Kaggle notebook/code submission**, not a manual upload of a local CSV file.

Local notebooks still write prediction CSVs to:

```text
submissions/*.csv
```

These files are ignored by Git and logged to MLflow for debugging/comparison, but they are not the final submission mechanism for this competition.

To submit to Kaggle:

1. Create or open a Kaggle notebook attached to the competition.
2. Copy the chosen experiment from `notebooks/01_bow_decision_tree_research.ipynb`.
3. Copy any helper functions used by that experiment directly into the Kaggle notebook. Do not assume Kaggle can import this repo's `helpers/` package.
4. Make sure the Kaggle notebook reads from Kaggle input paths, for example `/kaggle/input/competitions/contradictory-my-dear-watson/`.
5. The notebook must create `submission.csv` in the working directory.
6. Run the notebook end-to-end on Kaggle and submit the notebook output.

Notebook standard for team experiments:

```text
load data
build features
train model
evaluate on validation split
compute global and per-language metrics
predict Kaggle test
write submission.csv
log params/metrics/artifacts to MLflow when credentials are available
```

New experiment notebooks should stay close to the starter notebook structure, so results remain comparable and easy to review.

## 9. Promote A Model For FastAPI

Open DagsHub MLflow, choose a run, and copy its `run_id`.

Promote that run locally:

```bash
uv run python scripts/promote_model.py --run-id <mlflow_run_id>
```

This downloads the run's `model` artifact folder into:

```text
models/current/
```

`models/current` is ignored by Git. It is only the local model copy used by FastAPI.

## 10. Run FastAPI

Start API with Docker:

```bash
docker compose up --build api
```

Check health:

```bash
curl http://localhost:8000/health
```

Run prediction:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"premise":"A man is playing guitar.","hypothesis":"A person is making music.","language":"en"}'
```

FastAPI currently supports the baseline `sklearn_text_pair_vectorizer` serving format. Other model packages can be logged and promoted, but they need a matching loader in `app/backend/service.py` before they can be served.

## 11. Run Tests

```bash
uv run pytest
```

## 12. What Goes Where

```text
Git:
  code, helpers, notebooks without outputs, configs, docs, DVC metadata

DVC:
  raw Kaggle CSV files
  processed train/validation/test splits
  split metadata

MLflow:
  notebook experiment params
  validation metrics
  per-language metrics
  confusion matrices as JSON artifacts
  model artifacts: sklearn, TREE-G, transformer, or custom files
  submission.csv

Local only:
  .env
  .dvc/config.local
  .venv/
  submissions/*.csv
  models/current/
  mlruns/
```
