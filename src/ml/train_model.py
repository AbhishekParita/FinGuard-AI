"""Train a baseline model and optionally log the run to MLflow."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "generated" / "transactions.csv"
ARTIFACTS = ROOT / "artifacts"
SEED = 42


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pick_threshold(y_true, probabilities, default: float = 0.5) -> float:
    """Probability cut-off that maximises F1. Fraud is rare, so 0.5 is rarely the right choice."""
    if y_true.sum() == 0:
        return default
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.argmax(f1))])


def train() -> dict[str, float | str]:
    load_dotenv(ROOT / ".env", override=False)
    if not DATA.exists():
        raise FileNotFoundError("Run `python -m src.data.generate_data` first.")
    df = pd.read_csv(DATA)
    target = "is_synthetic_fraud"
    features = ["amount", "hour", "category", "transaction_type", "merchant"]
    X = df[features]
    y = df[target]
    if y.nunique() < 2:
        raise ValueError("Training data must contain both label classes.")

    numeric = ["amount", "hour"]
    categorical = ["category", "transaction_type", "merchant"]
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    model = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", RandomForestClassifier(
            # Shallow trees + minimum leaf size: fraud is rare, deep trees just memorise noise.
            n_estimators=300, max_depth=6, min_samples_leaf=20,
            class_weight="balanced", random_state=SEED, n_jobs=-1,
        )),
    ])

    # train / validation / test = 60 / 15 / 25. The decision threshold is chosen on the
    # validation split only, so the test metrics stay an honest estimate.
    X_rest, X_test, y_rest, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_rest, y_rest, test_size=0.20, random_state=SEED, stratify=y_rest)

    model.fit(X_train, y_train)
    threshold = pick_threshold(y_val, model.predict_proba(X_val)[:, 1])

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "threshold": threshold,
        "positive_rate": float(y_test.mean()),   # accuracy of always predicting "not fraud" is 1 - this
    }
    dataset_hash = sha256_file(DATA)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACTS / "fraud_risk_model.joblib"
    joblib.dump(model, model_path)
    metadata = {
        "dataset_file": DATA.name,
        "dataset_sha256": dataset_hash,
        "seed": SEED,
        "target": target,
        "features": features,
        "decision_threshold": threshold,
        "metrics": metrics,
        "note": "Synthetic demonstration labels only; not validated for real-world fraud detection."
    }
    (ARTIFACTS / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # If no remote URI is set, MLflow writes to the local mlruns directory.
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("finguard-ai-baseline")
    with mlflow.start_run():
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("random_state", SEED)
        mlflow.log_param("max_depth", 6)
        mlflow.log_param("min_samples_leaf", 20)
        mlflow.log_param("dataset_sha256", dataset_hash)
        mlflow.log_param("dataset_rows", len(df))
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.log_artifact(str(ARTIFACTS / "model_metadata.json"), artifact_path="metadata")
    print(json.dumps({"metrics": metrics, "dataset_sha256": dataset_hash, "model_path": str(model_path)}, indent=2))
    return metrics


if __name__ == "__main__":
    train()
