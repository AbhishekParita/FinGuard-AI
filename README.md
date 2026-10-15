# FinGuard AI

A small local financial intelligence demo built entirely on synthetic data. Python handles data access, calculations, validation, and ML predictions. A Node.js server hosts the browser UI and starts the local FastAPI backend.

## Local features

- Overview of client activity, portfolio value, and model review flags
- Searchable transaction list with saved-model fraud probabilities
- Client profile and synthetic KYC fields
- Portfolio holdings, cost basis, market value, and unrealized change
- Generated instrument prices and synthetic news
- Natural-language question box with transaction, portfolio, risk, profile, and market specialists
- SHAP feature contributions for the highest-risk transaction
- Optional Agno + Hugging Face explanation; deterministic local answers work without a token
- Local SQLite demo login and saved assistant history
- Local MLflow tracking and pytest-based GitHub Actions CI

All data, prices, news, labels, and profiles are generated for demonstration. The local app binds to `127.0.0.1`. The demo login is a basic local gate, not real-user authorization; do not use real financial or customer records.

## Start locally on Windows

Install Python 3.11+ and Node.js 20+. From PowerShell in this folder:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.data.generate_data --seed 42 --clients 250 --transactions 10000
.\.venv\Scripts\python.exe -m src.data.validate_data
.\.venv\Scripts\python.exe -m src.ml.train_model
npm.cmd start
```

Open <http://127.0.0.1:5173>. `npm.cmd start` serves the UI and starts FastAPI on `127.0.0.1:8000`; stop it with Ctrl+C. The Node frontend uses built-in Node modules, so no `npm install` is needed.

Sign in with `demo` / `FinGuardDemo!`. You can set `FINGUARD_DEMO_USER` and `FINGUARD_DEMO_PASSWORD` in `.env` before the first launch to choose another demo login. SQLite stores the local account, temporary login session, and assistant chat history in `data/finguard_demo.sqlite3`; this file is ignored by Git. The financial demo records continue to come from the generated local CSV files.

## Optional Hugging Face explanations

The local assistant routes requests and gathers evidence with deterministic Python code. Agno is an installed open-source library and needs no separate Agno account or token. Hugging Face is optional: the dashboard works without a token, using deterministic local answers. To enable a hosted Hugging Face model for grounded wording, create an optional Hugging Face token and keep it in the ignored local `.env` file (never paste it into chat):

```powershell
$env:HF_TOKEN = "your-token"
$env:HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"
npm.cmd start
```

Or add `HF_TOKEN` and `HF_MODEL` to your ignored local `.env` file. The Node launcher reads that file without printing its contents. When enabled, the selected client's synthetic evidence is sent to the configured Hugging Face inference service for explanation. Hugging Face hosted inference has a limited free allowance that may change; it is not guaranteed to be unlimited free. Without a token, the assistant returns the locally calculated answer without an LLM call. Agno telemetry is disabled by default.

## Model and experiment tracking

`python -m src.ml.train_model` saves `artifacts/fraud_risk_model.joblib` and `artifacts/model_metadata.json`. FastAPI uses those saved files for inference; the training dataset is not read again for each prediction. Training runs are recorded locally under `mlruns/` unless `MLFLOW_TRACKING_URI` is configured.

For optional DagsHub tracking, add `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, and `MLFLOW_TRACKING_PASSWORD` to the ignored local `.env` file before training. Keep credentials local; they are never needed for the local MLflow workflow.

The Random Forest is trained on synthetic labels and is not a validated fraud detector. SHAP values describe the model's feature contributions; they do not prove fraud or explain a real-world cause.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The generator writes all CSV inputs to `data/generated/`. Re-run the generator and training command whenever you want to reset the synthetic data and model.
