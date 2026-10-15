"""Deterministic analytics over the local synthetic financial datasets."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data" / "generated"


def _read(name: str, required: tuple[str, ...] = ()) -> pd.DataFrame:
    path = DATA / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path.name}. Run python -m src.data.generate_data first.")
    frame = pd.read_csv(path)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {', '.join(sorted(missing))}")
    return frame


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    clients = _read("clients.csv", ("client_id", "age", "annual_income", "credit_score", "risk_profile"))
    transactions = _read("transactions.csv", ("transaction_id", "client_id", "transaction_date", "merchant", "category", "amount", "hour", "transaction_type", "is_synthetic_fraud"))
    return clients, transactions


def list_clients() -> list[dict[str, str]]:
    clients, _ = load_data()
    names = {}
    path = DATA / "book_clients.csv"
    if path.exists():
        book_clients = pd.read_csv(path, usecols=lambda col: col in {"client_id", "name"})
        if "name" in book_clients:
            names = dict(zip(book_clients.client_id, book_clients.name))
    return [{"client_id": str(row.client_id), "name": str(names.get(row.client_id, row.client_id))}
            for row in clients.itertuples(index=False)]


def client_profile(client_id: str) -> dict:
    clients, _ = load_data()
    profile = clients.loc[clients.client_id == client_id]
    if profile.empty:
        raise KeyError(client_id)
    row = profile.iloc[0]
    result = {
        "client_id": client_id,
        "age": int(row.age),
        "annual_income": float(row.annual_income),
        "credit_score": int(row.credit_score),
        "risk_profile": str(row.risk_profile),
    }
    path = DATA / "book_clients.csv"
    if path.exists():
        book = pd.read_csv(path, usecols=lambda col: col in {
            "client_id", "name", "kyc_status", "risk_profile", "annual_income_band", "account_opened"
        })
        match = book.loc[book.client_id == client_id]
        if not match.empty:
            item = match.iloc[0]
            result["account"] = {
                "name": str(item.get("name", "")),
                "kyc_status": str(item.get("kyc_status", "unknown")),
                "investment_risk_profile": str(item.get("risk_profile", "unknown")),
                "annual_income_band": str(item.get("annual_income_band", "unknown")),
                "account_opened": str(item.get("account_opened", "")),
            }
    return result


def client_summary(client_id: str) -> dict:
    profile = client_profile(client_id)
    _, transactions = load_data()
    rows = transactions.loc[transactions.client_id == client_id]
    return {
        **profile,
        "transaction_count": int(len(rows)),
        "total_transaction_amount": round(float(rows.amount.sum()), 2),
        "average_transaction_amount": round(float(rows.amount.mean()), 2) if len(rows) else 0.0,
        "synthetic_flag_count": int(rows.is_synthetic_fraud.sum()),
        "categories": {str(k): int(v) for k, v in rows.category.value_counts().items()},
    }


def list_transactions(client_id: str, limit: int = 25, category: str | None = None) -> list[dict]:
    client_profile(client_id)
    _, rows = load_data()
    rows = rows.loc[rows.client_id == client_id].copy()
    if category:
        rows = rows.loc[rows.category.str.casefold() == category.casefold()]
    rows = rows.sort_values(["transaction_date", "transaction_id"], ascending=False).head(max(1, min(limit, 100)))
    return rows.drop(columns=["is_synthetic_fraud"]).to_dict(orient="records")


def transaction_aggregate(client_id: str, start_date: str | None = None, end_date: str | None = None) -> dict:
    client_profile(client_id)
    _, rows = load_data()
    rows = rows.loc[rows.client_id == client_id].copy()
    rows["transaction_date"] = pd.to_datetime(rows.transaction_date)
    if start_date:
        rows = rows.loc[rows.transaction_date >= pd.Timestamp(start_date)]
    if end_date:
        rows = rows.loc[rows.transaction_date <= pd.Timestamp(end_date)]
    return {
        "client_id": client_id,
        "start_date": start_date,
        "end_date": end_date,
        "transaction_count": int(len(rows)),
        "total_amount": round(float(rows.amount.sum()), 2),
        "average_amount": round(float(rows.amount.mean()), 2) if len(rows) else 0.0,
        "category_totals": {str(k): round(float(v), 2) for k, v in rows.groupby("category").amount.sum().items()},
    }


def portfolio_summary(client_id: str) -> dict:
    positions = _read("book_positions.csv", ("client_id", "symbol", "quantity", "avg_cost_usd", "market_value_usd"))
    instruments = _read("instruments.csv", ("symbol", "sector", "industry", "currency", "listed_on"))
    held = positions.loc[positions.client_id == client_id].merge(instruments, on="symbol", how="left")
    if held.empty:
        # Confirm the client exists before returning an empty portfolio.
        client_profile(client_id)
    held["cost_basis_usd"] = held.quantity * held.avg_cost_usd
    held["gain_loss_usd"] = held.market_value_usd - held.cost_basis_usd
    held["gain_loss_pct"] = held.apply(
        lambda row: (row.gain_loss_usd / row.cost_basis_usd * 100) if row.cost_basis_usd else 0.0, axis=1
    )
    positions_out = [{
        "symbol": str(row.symbol), "sector": str(row.sector), "industry": str(row.industry),
        "quantity": round(float(row.quantity), 4), "avg_cost_usd": round(float(row.avg_cost_usd), 2),
        "market_value_usd": round(float(row.market_value_usd), 2),
        "cost_basis_usd": round(float(row.cost_basis_usd), 2),
        "gain_loss_usd": round(float(row.gain_loss_usd), 2),
        "gain_loss_pct": round(float(row.gain_loss_pct), 2),
    } for row in held.itertuples(index=False)]
    total_value = sum(row["market_value_usd"] for row in positions_out)
    total_cost = sum(row["cost_basis_usd"] for row in positions_out)
    last_prices = _latest_prices()
    return {
        "client_id": client_id,
        "as_of": max(last_prices.values(), key=lambda item: item["date"])["date"] if last_prices else None,
        "position_count": len(positions_out),
        "total_value_usd": round(total_value, 2),
        "total_cost_usd": round(total_cost, 2),
        "gain_loss_usd": round(total_value - total_cost, 2),
        "gain_loss_pct": round((total_value - total_cost) / total_cost * 100, 2) if total_cost else 0.0,
        "positions": positions_out,
    }


def _latest_prices() -> dict[str, dict]:
    prices = _read("market_prices.csv", ("symbol", "date", "close"))
    prices["date"] = pd.to_datetime(prices.date)
    latest = prices.sort_values("date").groupby("symbol", as_index=False).tail(1)
    return {str(row.symbol): {"price": round(float(row.close), 2), "date": row.date.strftime("%Y-%m-%d")}
            for row in latest.itertuples(index=False)}


def market_snapshot(symbol: str | None = None) -> dict:
    instruments = _read("instruments.csv", ("symbol", "sector", "industry", "currency", "listed_on"))
    prices = _latest_prices()
    if symbol:
        instruments = instruments.loc[instruments.symbol.str.casefold() == symbol.casefold()]
    quotes = [{
        "symbol": str(row.symbol), "sector": str(row.sector), "industry": str(row.industry),
        "currency": str(row.currency), "exchange": str(row.listed_on),
        "close": prices.get(row.symbol, {}).get("price"),
        "date": prices.get(row.symbol, {}).get("date"),
    } for row in instruments.itertuples(index=False)]
    news = _read("news.csv", ("id", "date", "symbol", "headline", "body", "source"))
    if symbol:
        news = news.loc[news.symbol.str.casefold() == symbol.casefold()]
    news = news.sort_values(["date", "id"], ascending=False).head(12)
    return {"quotes": quotes, "news": news.to_dict(orient="records")}
