"""Validate generated dataset shape and integrity."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "generated"

# Fraud is rare, but a model needs enough positives to learn from. Outside this band the
# dataset is either too easy / unrealistic (too many) or unlearnable (too few).
MIN_FRAUD_RATE = 0.02
MAX_FRAUD_RATE = 0.15
MIN_FRAUD_ROWS = 100


def validate() -> list[str]:
    errors: list[str] = []
    clients_path = DATA / "clients.csv"
    tx_path = DATA / "transactions.csv"
    if not clients_path.exists() or not tx_path.exists():
        return ["Missing generated CSVs. Run src.data.generate_data first."]

    clients = pd.read_csv(clients_path)
    tx = pd.read_csv(tx_path)
    required_clients = {"client_id", "age", "annual_income", "credit_score", "risk_profile"}
    required_tx = {
        "transaction_id", "client_id", "transaction_date", "merchant",
        "category", "amount", "hour", "transaction_type", "is_synthetic_fraud"
    }
    if not required_clients.issubset(clients.columns):
        errors.append(f"Missing client columns: {sorted(required_clients - set(clients.columns))}")
    if not required_tx.issubset(tx.columns):
        errors.append(f"Missing transaction columns: {sorted(required_tx - set(tx.columns))}")
    if errors:                       # the checks below need these columns
        return errors
    if clients["client_id"].duplicated().any():
        errors.append("client_id values must be unique")
    if tx["transaction_id"].duplicated().any():
        errors.append("transaction_id values must be unique")
    if not tx["client_id"].isin(clients["client_id"]).all():
        errors.append("Transactions contain unknown client IDs")
    if (tx["amount"] < 0).any():
        errors.append("Transaction amounts must be non-negative")
    if not tx["hour"].between(0, 23).all():
        errors.append("hour must be between 0 and 23")
    if not tx["is_synthetic_fraud"].isin([0, 1]).all():
        errors.append("is_synthetic_fraud must be binary")
    else:
        rate = tx["is_synthetic_fraud"].mean()
        positives = int(tx["is_synthetic_fraud"].sum())
        if not MIN_FRAUD_RATE <= rate <= MAX_FRAUD_RATE:
            errors.append(f"fraud rate {rate:.2%} outside {MIN_FRAUD_RATE:.0%}-{MAX_FRAUD_RATE:.0%}")
        if positives < MIN_FRAUD_ROWS:
            errors.append(f"only {positives} fraud rows; need at least {MIN_FRAUD_ROWS} to train on")
    if not clients["credit_score"].between(300, 850).all():
        errors.append("credit_score must be between 300 and 850")
    return errors


def main() -> None:
    errors = validate()
    if errors:
        raise SystemExit("\n".join(f"ERROR: {e}" for e in errors))
    tx = pd.read_csv(DATA / "transactions.csv")
    print(f"Data validation passed. Fraud rate: {tx['is_synthetic_fraud'].mean():.2%} "
          f"({int(tx['is_synthetic_fraud'].sum())} of {len(tx):,} rows)")


if __name__ == "__main__":
    main()
