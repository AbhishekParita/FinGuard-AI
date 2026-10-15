"""Load the saved classifier and score transaction feature rows."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "artifacts" / "fraud_risk_model.joblib"
METADATA_PATH = ROOT / "artifacts" / "model_metadata.json"


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError("Trained model unavailable. Run python -m src.ml.train_model first.")
    return joblib.load(MODEL_PATH), json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def score_transactions(rows: list[dict]) -> tuple[list[dict], dict]:
    model, metadata = load_model()
    result = [dict(row) for row in rows]
    if result:
        probabilities = model.predict_proba(pd.DataFrame(result)[metadata["features"]])[:, 1]
    else:
        probabilities = []
    threshold = float(metadata["decision_threshold"])
    for row, probability in zip(result, probabilities):
        row["fraud_probability"] = round(float(probability), 4)
        row["predicted_fraud"] = bool(probability >= threshold)
    return result, {
        "name": "Random Forest fraud risk model",
        "decision_threshold": threshold,
        "note": metadata["note"],
    }


def client_risk_summary(client_id: str) -> dict:
    from src.tools.financial import client_profile, list_transactions

    profile = client_profile(client_id)
    rows = list_transactions(client_id, limit=100)
    scored, model = score_transactions(rows)
    return {
        "client_id": client_id,
        "credit_score": profile["credit_score"],
        "credit_risk_profile": profile["risk_profile"],
        "transaction_count_scored": len(scored),
        "model_flagged_count": sum(row["predicted_fraud"] for row in scored),
        "highest_risk_transactions": sorted(scored, key=lambda row: row["fraud_probability"], reverse=True)[:5],
        "model": model,
    }


def explain_transaction(row: dict) -> dict:
    """Return SHAP contributions for the saved pipeline's positive class."""
    import shap

    model, metadata = load_model()
    features = metadata["features"]
    values = pd.DataFrame([{name: row[name] for name in features}])
    pipeline = model.named_steps["preprocess"]
    classifier = model.named_steps["classifier"]
    transformed = pipeline.transform(values)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    explainer = shap.TreeExplainer(classifier)
    raw_values = explainer.shap_values(transformed)
    if isinstance(raw_values, list):
        positive_values = np.asarray(raw_values[1])[0]
    else:
        raw_values = np.asarray(raw_values)
        positive_values = raw_values[0, :, 1] if raw_values.ndim == 3 else raw_values[0]
    feature_names = pipeline.get_feature_names_out()
    contributions = [{
        "feature": str(name).replace("num__", "").replace("cat__", ""),
        "impact": round(float(value), 5),
    } for name, value in zip(feature_names, positive_values)]
    contributions.sort(key=lambda item: abs(item["impact"]), reverse=True)
    expected = np.asarray(explainer.expected_value).reshape(-1)
    baseline = float(expected[1] if len(expected) > 1 else expected[0])
    probability = float(model.predict_proba(values)[0, 1])
    return {
        "transaction_id": row.get("transaction_id"),
        "probability": round(probability, 4),
        "baseline": round(baseline, 5),
        "contributions": contributions[:8],
        "note": "SHAP values are additive model explanations; they do not establish actual fraud.",
    }
