"""Generate fully synthetic client, transaction, market and brokerage data.

Standalone: needs no input files. Everything is drawn from a seeded numpy RNG.

Outputs (data/generated/ by default)
  clients.csv, transactions.csv          <- your original generator, unchanged
  instruments.csv, market_prices.csv, news.csv
  book_clients.csv, book_transactions.csv, book_positions.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "generated"

AS_OF = pd.Timestamp("2026-07-31")
FEE_RATE = 0.0025          # brokerage fee on buys / sells
WHT_RATE = 0.25            # withholding tax on dividends
PLATFORM_FEE = 3.19        # monthly subscription fee (USD)


# --------------------------------------------------------------------------- #
# 1. ORIGINAL generator (unchanged logic, so existing code keeps working)
# --------------------------------------------------------------------------- #
def generate(seed: int = 42, clients: int = 250, transactions: int = 10_000) -> tuple[pd.DataFrame, pd.DataFrame]:
    if clients < 10:
        raise ValueError("clients must be at least 10")
    if transactions < clients:
        raise ValueError("transactions must be at least the number of clients")

    rng = np.random.default_rng(seed)
    client_ids = [f"CL{n:05d}" for n in range(1, clients + 1)]
    profiles = pd.DataFrame({
        "client_id": client_ids,
        "age": rng.integers(18, 80, size=clients),
        "annual_income": np.round(rng.lognormal(mean=10.7, sigma=0.55, size=clients), 2),
        "credit_score": rng.integers(300, 851, size=clients),
        "risk_profile": rng.choice(["low", "medium", "high"], size=clients, p=[0.45, 0.4, 0.15]),
    })

    tx_client_idx = rng.integers(0, clients, size=transactions)
    amounts = np.round(rng.lognormal(mean=3.2, sigma=1.0, size=transactions), 2)
    categories = rng.choice(
        ["groceries", "utilities", "shopping", "travel", "cash", "transfer", "dining"],
        size=transactions,
        p=[0.22, 0.12, 0.20, 0.08, 0.08, 0.18, 0.12],
    )
    hours = rng.integers(0, 24, size=transactions)
    # Synthetic demonstration label: unusual amounts, cash/transfer categories and
    # late-night hours raise the fraud probability (and compound when combined), but
    # never determine it. Tuned for ~4-5% positives with enough learnable signal that a
    # good model reaches ROC-AUC ~0.85-0.9 (the labels stay noisy by design).
    risky_cat = np.isin(categories, ["cash", "transfer"])
    night = (hours <= 4) | (hours >= 23)
    logit = (
        -5.5
        + 2.2 * (amounts > 100)
        + 1.4 * (amounts > 300)
        + 1.0 * (amounts > 1000)
        + 1.9 * risky_cat
        + 1.7 * night
        + 1.2 * (risky_cat & night)
        + 0.9 * (categories == "travel")
    )

    probs = 1 / (1 + np.exp(-logit))
    labels = rng.binomial(1, probs)
    dates = pd.Timestamp("2024-01-01") + pd.to_timedelta(
        rng.integers(0, 365 * 3, size=transactions), unit="D"
    )

    tx = pd.DataFrame({
        "transaction_id": [f"TX{n:08d}" for n in range(1, transactions + 1)],
        "client_id": np.array(client_ids)[tx_client_idx],
        "transaction_date": dates.strftime("%Y-%m-%d"),
        "merchant": rng.choice(
            ["Market Square", "Metro Utilities", "Northstar Travel", "Everyday Store", "City Dining", "ATM Network"],
            size=transactions,
        ),
        "category": categories,
        "amount": amounts,
        "hour": hours,
        "transaction_type": rng.choice(["debit", "credit"], size=transactions, p=[0.82, 0.18]),
        "is_synthetic_fraud": labels,
    })
    return profiles, tx


# --------------------------------------------------------------------------- #
# 2. Market data: instruments, monthly prices, news
# --------------------------------------------------------------------------- #
# symbol, sector, industry, listed_on, starting price, behaviour group
INSTRUMENTS = [
    ("AAPL", "Information Technology", "Consumer Electronics", "NASDAQ", 190.0, "growth"),
    ("AMD", "Information Technology", "Semiconductors", "NASDAQ", 150.0, "growth"),
    ("AMZN", "Consumer Discretionary", "Internet Retail", "NASDAQ", 185.0, "growth"),
    ("GOOG", "Communication Services", "Interactive Media", "NASDAQ", 165.0, "growth"),
    ("INTC", "Information Technology", "Semiconductors", "NASDAQ", 36.0, "growth"),
    ("JPM", "Financials", "Diversified Banks", "NYSE", 200.0, "defensive"),
    ("KO", "Consumer Staples", "Soft Drinks", "NYSE", 62.0, "defensive"),
    ("META", "Communication Services", "Interactive Media", "NASDAQ", 520.0, "growth"),
    ("MSFT", "Information Technology", "Software", "NASDAQ", 430.0, "growth"),
    ("NFLX", "Communication Services", "Entertainment", "NASDAQ", 640.0, "growth"),
    ("NVDA", "Information Technology", "Semiconductors", "NASDAQ", 900.0, "growth"),
    ("QQQ", "Diversified", "Broad Market ETF", "NASDAQ", 470.0, "etf"),
    ("TSLA", "Consumer Discretionary", "Automobiles", "NASDAQ", 230.0, "growth"),
    ("VOO", "Diversified", "Broad Market ETF", "NYSE", 500.0, "etf"),
]
MONTHLY_VOL = {"growth": 0.045, "defensive": 0.033, "etf": 0.028}

NEWS_TEMPLATES = [
    ("{s} appoints a new finance chief", "The new appointee starts next quarter and the previous officer will assist with the handover."),
    ("{s} says supply bottlenecks are easing", "Delivery lead times improved during the period, though no volume figure was given."),
    ("{s} raises its outlook for the year", "Management cited demand running ahead of plan and left capital spending guidance unchanged."),
    ("{s} announces a buyback", "The board authorised repurchases over the next four quarters, funded from operating cash flow."),
    ("Analysts divided on {s} after results", "Some see a turning point, others a one-off. No consensus revision has been issued."),
    ("{s} adds production capacity", "The new capacity is due online next year and sits within previously announced capital plans."),
    ("Routine regulatory review covers {s}", "The review is sector-wide. The company is cooperating and expects no material impact."),
    ("{s} buys a small business", "Terms were not disclosed and the deal is not expected to affect current-year earnings."),
]


def generate_market(seed: int = 42, months: int = 25, news_items: int = 40):
    rng = np.random.default_rng(seed + 1)
    dates = pd.date_range("2024-07-01", periods=months, freq="MS")

    instruments = pd.DataFrame(
        [(s, sec, ind, "USD", ex) for s, sec, ind, ex, _, _ in INSTRUMENTS],
        columns=["symbol", "sector", "industry", "currency", "listed_on"],
    )

    market_shock = rng.normal(0.004, 0.015, size=months - 1)       # shared factor -> realistic correlation
    rows = []
    for sym, _, _, _, base, kind in INSTRUMENTS:
        idio = rng.normal(0.0, MONTHLY_VOL[kind], size=months - 1)
        rets = market_shock + idio
        closes = base * np.concatenate([[1.0], np.cumprod(1 + rets)])
        rows += [(sym, d, round(float(c), 2)) for d, c in zip(dates, closes)]
    prices = pd.DataFrame(rows, columns=["symbol", "date", "close"])

    tpl = rng.integers(0, len(NEWS_TEMPLATES), size=news_items)
    syms = rng.choice([i[0] for i in INSTRUMENTS], size=news_items)
    days = rng.integers(0, (dates[-1] - dates[0]).days, size=news_items)
    news = pd.DataFrame({
        "id": [f"news_{3000 + n}" for n in range(news_items)],
        "date": (dates[0] + pd.to_timedelta(days, unit="D")).strftime("%Y-%m-%d"),
        "symbol": syms,
        "headline": [NEWS_TEMPLATES[t][0].format(s=s) for t, s in zip(tpl, syms)],
        "body": [NEWS_TEMPLATES[t][1] for t in tpl],
        "source": "Synthetic Newswire",
    }).sort_values(["date", "id"]).reset_index(drop=True)
    return instruments, prices, news


# --------------------------------------------------------------------------- #
# 3. Brokerage book: clients, ledger, position snapshot
# --------------------------------------------------------------------------- #
FIRST_M = ["Aditya", "Karan", "Nikhil", "Vivek", "Manish", "Yash", "Devansh", "Kabir", "Mohit", "Pranav",
           "Utkarsh", "Lakshay", "Neeraj", "Chirag", "Dhruv", "Kunal", "Omkar", "Rajat", "Tanmay", "Vikram"]
FIRST_F = ["Divya", "Kavya", "Neha", "Riya", "Tanvi", "Isha", "Sanya", "Nandini", "Prerna", "Aditi",
           "Bhavna", "Charu", "Disha", "Gayatri", "Harini", "Jhanvi", "Komal", "Lavanya", "Mansi", "Sakshi"]
LAST = ["Agarwal", "Chatterjee", "Desai", "Iyer", "Joshi", "Khanna", "Mehta", "Naidu", "Patil", "Rao",
        "Saxena", "Tiwari", "Venkatesh", "Yadav", "Bose", "Chopra", "Dutta", "Gupta", "Hegde", "Kamath"]
CITIES = ["Mumbai", "Pune", "Chennai", "Bengaluru", "Kolkata", "Jaipur", "Ahmedabad", "Hyderabad", "Kochi", "Gurugram"]
STREETS = ["MG Road", "Residency Road", "Lake View", "Sector 21", "Park Avenue", "Station Road"]
BANKS = ["HDFC Bank", "ICICI Bank", "Axis Bank", "Kotak Bank"]
INCOME_BANDS = ["5-10 LPA", "10-25 LPA", "25-50 LPA", "50+ LPA"]
# preference weights by instrument group for each risk profile
PREF = {
    "conservative": {"defensive": 3.0, "etf": 3.0, "growth": 1.0},
    "moderate": {"defensive": 2.0, "etf": 2.0, "growth": 2.0},
    "aggressive": {"defensive": 0.5, "etf": 1.0, "growth": 4.0},
}


def _digits(rng, n: int) -> str:
    return "".join(rng.choice(list("0123456789"), size=n))


def _letters(rng, n: int) -> str:
    return "".join(rng.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), size=n))


def generate_book(prices: pd.DataFrame, seed: int = 42, clients: int = 250, transactions: int = 10_000):
    """Clients + ledger + position snapshot. `transactions` is the approximate ledger size (incl. fee rows)."""
    rng = np.random.default_rng(seed + 2)
    month_px = {(r.symbol, r.date.strftime("%Y-%m")): r.close for r in prices.itertuples()}
    last_px = prices.sort_values("date").groupby("symbol")["close"].last().to_dict()
    kind_of = {i[0]: i[5] for i in INSTRUMENTS}
    symbols = [i[0] for i in INSTRUMENTS]

    # ---- client table -----------------------------------------------------
    risk = rng.choice(["conservative", "moderate", "aggressive"], size=clients, p=[0.40, 0.15, 0.45])
    female = rng.random(clients) < 0.5
    opened = pd.Timestamp("2024-07-01") + pd.to_timedelta(rng.integers(0, 365, size=clients), unit="D")
    book_clients = pd.DataFrame({
        "client_id": [f"CL{n:05d}" for n in range(1, clients + 1)],
        "name": [f"{rng.choice(FIRST_F if f else FIRST_M)} {rng.choice(LAST)}" for f in female],
        # 4th PAN character is deliberately invalid so these can never match a real PAN
        "pan": [_letters(rng, 3) + "Z" + _letters(rng, 1) + _digits(rng, 4) + _letters(rng, 1) for _ in range(clients)],
        "kyc_status": "verified",
        "risk_profile": risk,
        "date_of_birth": (pd.Timestamp("1960-01-01") + pd.to_timedelta(rng.integers(0, 40 * 365, size=clients), unit="D")).strftime("%Y-%m-%d"),
        "address": [f"{rng.integers(10, 900)} {rng.choice(STREETS)}, {rng.choice(CITIES)}" for _ in range(clients)],
        "annual_income_band": rng.choice(INCOME_BANDS, size=clients, p=[0.24, 0.32, 0.28, 0.16]),
        "bank": rng.choice(BANKS, size=clients),
        "bank_account_number": ["999" + _digits(rng, 11) for _ in range(clients)],   # 999 prefix = fake
        "ifsc": ["VALU0" + _digits(rng, 6) for _ in range(clients)],                 # VALU0 bank code does not exist
        "broker_ref": ["VTX-" + _digits(rng, 8) for _ in range(clients)],
        "account_opened": opened.strftime("%Y-%m-%d"),
    })

    # ---- ledger -----------------------------------------------------------
    rows, positions, lrs_used = [], [], set()
    # heavy-tailed activity: ~16% of clients are active traders with ~25x the events of a typical client
    weight = np.where(rng.random(clients) < 0.16, 25.0, 1.0)
    target_events = transactions * weight / weight.sum()

    for ci, cid in enumerate(book_clients["client_id"]):
        start = opened[ci]
        w = np.array([PREF[risk[ci]][kind_of[s]] for s in symbols], dtype=float)
        prefs = list(rng.choice(symbols, size=int(rng.integers(3, 6)), replace=False, p=w / w.sum()))

        # subscription fee: only a client's first few months, capped relative to activity (keeps fees ~2-3% of the ledger)
        fee_dates = pd.date_range(start + pd.offsets.MonthBegin(1), AS_OF, freq="MS")[: min(int(rng.integers(3, 14)), 1 + int(0.1 * target_events[ci]))]
        n = max(8, int(rng.poisson(target_events[ci])) - len(fee_dates))
        offs = np.sort(rng.integers(1, max(2, (AS_OF - start).days), size=n - 1))
        kinds = rng.choice(["buy", "sell", "deposit", "withdrawal", "dividend"], size=n - 1, p=[0.42, 0.22, 0.14, 0.12, 0.10])
        events = [(start, "deposit")]
        events += [(start + pd.Timedelta(days=int(o)), k) for o, k in zip(offs, kinds)]
        events += [(d, "fee") for d in fee_dates]
        events.sort(key=lambda e: e[0])            # stable: opening deposit stays first

        cash, hold, cost = 0.0, {}, {}
        for d, kind in events:
            month = d.strftime("%Y-%m")
            held = [s for s, q in hold.items() if q > 1e-4]
            if kind == "sell" and not held:
                kind = "buy"
            if kind == "dividend" and not held:
                kind = "deposit"
            if kind in ("buy", "withdrawal") and cash < 200:
                kind = "deposit"
            row = {"client_id": cid, "transaction_date": d.strftime("%Y-%m-%d"), "transaction_type": kind}

            if kind == "fee":
                if cash < PLATFORM_FEE:
                    continue
                cash -= PLATFORM_FEE
                row.update(amount_usd=PLATFORM_FEE, description="platform subscription fee")

            elif kind == "deposit":
                fx = round(float(np.clip(84.0 + rng.normal(0, 1.1), 82.0, 86.5)), 4)   # round FX first
                usd = float(rng.lognormal(8.6, 0.6))
                inr = round(usd * fx / 10) * 10
                usd = round(inr / fx, 2)
                while (lrs := f"LRS26-{rng.integers(100000, 999999)}") in lrs_used:
                    pass
                lrs_used.add(lrs)
                cash += usd
                row.update(amount_inr=float(inr), fx_rate=fx, amount_usd=usd, lrs_ref=lrs)

            elif kind == "withdrawal":
                amt = round(cash * rng.uniform(0.10, 0.50), 2)
                cash -= amt
                row.update(amount_usd=amt, description="linked bank account")

            elif kind == "buy":
                sym = str(rng.choice(prefs))
                price = max(1.0, round(month_px.get((sym, month), last_px[sym]) * (1 + rng.normal(0, 0.03)), 2))
                qty = round(cash / (1 + FEE_RATE) * rng.uniform(0.2, 0.8) / price, 4)
                gross = round(qty * price, 2)
                fee = round(gross * FEE_RATE, 2)
                cash -= gross + fee
                hold[sym] = hold.get(sym, 0.0) + qty
                cost[sym] = cost.get(sym, 0.0) + gross
                row.update(symbol=sym, quantity=qty, price_usd=price, gross_usd=gross, fees_usd=fee, net_usd=round(gross + fee, 2))

            elif kind == "sell":
                sym = str(rng.choice(held))
                price = max(1.0, round(month_px.get((sym, month), last_px[sym]) * (1 + rng.normal(0, 0.03)), 2))
                full = rng.random() < 0.25
                qty = round(hold[sym], 4) if full else round(hold[sym] * rng.uniform(0.10, 0.90), 4)
                qty = min(qty, hold[sym])
                gross = round(qty * price, 2)
                fee = round(gross * FEE_RATE, 2)
                cash += gross - fee
                cost[sym] *= max(0.0, 1 - qty / hold[sym])
                hold[sym] = round(hold[sym] - qty, 4)
                row.update(symbol=sym, quantity=qty, price_usd=price, gross_usd=gross, fees_usd=fee, net_usd=round(gross - fee, 2))

            else:  # dividend
                sym = str(rng.choice(held))
                price = month_px.get((sym, month), last_px[sym])
                gross = max(0.01, round(hold[sym] * price * rng.uniform(0.002, 0.006), 2))
                wht = round(gross * WHT_RATE, 2)
                cash += gross - wht
                row.update(symbol=sym, gross_usd=gross, withholding_tax_usd=wht, net_usd=round(gross - wht, 2))

            rows.append(row)

        for sym, q in hold.items():
            if q > 1e-4:
                positions.append({
                    "client_id": cid, "symbol": sym, "quantity": round(q, 4),
                    "avg_cost_usd": round(cost[sym] / q, 2),
                    "market_value_usd": round(q * last_px[sym], 2),
                })

    cols = ["transaction_id", "client_id", "transaction_date", "transaction_type", "symbol", "quantity", "price_usd",
            "gross_usd", "fees_usd", "net_usd", "withholding_tax_usd", "amount_inr", "fx_rate", "amount_usd",
            "lrs_ref", "description"]
    ledger = pd.DataFrame(rows).sort_values("transaction_date", kind="stable").reset_index(drop=True)
    ledger.insert(0, "transaction_id", [f"TX{n:08d}" for n in range(1, len(ledger) + 1)])
    ledger = ledger.reindex(columns=cols)
    return book_clients, ledger, pd.DataFrame(positions)


def validate_book(ledger: pd.DataFrame, positions: pd.DataFrame) -> None:
    """Raise AssertionError if the ledger breaks a basic accounting rule."""
    t = ledger
    trade = t["transaction_type"].isin(["buy", "sell"])
    assert (abs(t.loc[trade, "quantity"] * t.loc[trade, "price_usd"] - t.loc[trade, "gross_usd"]) <= 0.01).all(), "gross != qty*price"
    assert (abs(t.loc[trade, "fees_usd"] - t.loc[trade, "gross_usd"] * FEE_RATE) <= 0.006).all(), "fee rate"
    div = t["transaction_type"] == "dividend"
    assert (abs(t.loc[div, "withholding_tax_usd"] - t.loc[div, "gross_usd"] * WHT_RATE) <= 0.01).all(), "withholding rate"
    dep = t["transaction_type"] == "deposit"
    assert (abs(t.loc[dep, "amount_inr"] / t.loc[dep, "fx_rate"] - t.loc[dep, "amount_usd"]) <= 0.02).all(), "INR/FX != USD"

    sign = t["transaction_type"].map({"deposit": 1, "sell": 1, "dividend": 1, "withdrawal": -1, "fee": -1, "buy": -1})
    flow = np.where(t["transaction_type"].isin(["buy", "sell", "dividend"]), t["net_usd"], t["amount_usd"])
    cash = (sign * flow).groupby(t["client_id"]).cumsum()
    assert cash.min() >= -0.01, f"negative cash ({cash.min():.2f})"

    q = t["quantity"].where(t["transaction_type"] == "buy", -t["quantity"]).where(trade)
    held = q.groupby([t["client_id"], t["symbol"]]).cumsum()
    assert held.min() >= -1e-3, "sold more than held"

    net = q.groupby([t["client_id"], t["symbol"]]).sum()
    snap = positions.set_index(["client_id", "symbol"])["quantity"]
    net = net[net.abs() > 1e-3]
    assert np.allclose(net.reindex(snap.index.union(net.index)).fillna(0),
                       snap.reindex(snap.index.union(net.index)).fillna(0), atol=1e-3), "positions != trades"


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clients", type=int, default=250)
    parser.add_argument("--transactions", type=int, default=10_000)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--legacy-only", action="store_true", help="write only clients.csv / transactions.csv")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    profiles, tx = generate(args.seed, args.clients, args.transactions)
    profiles.to_csv(args.out / "clients.csv", index=False)
    tx.to_csv(args.out / "transactions.csv", index=False)
    print(f"Wrote {len(profiles):,} synthetic clients and {len(tx):,} transactions to {args.out}")
    print(f"Synthetic positive-label rate: {tx['is_synthetic_fraud'].mean():.2%}")
    if args.legacy_only:
        return

    instruments, prices, news = generate_market(args.seed)
    book_clients, ledger, positions = generate_book(prices, args.seed, args.clients, args.transactions)
    validate_book(ledger, positions)

    instruments.to_csv(args.out / "instruments.csv", index=False)
    prices.assign(date=prices["date"].dt.strftime("%Y-%m-%d")).to_csv(args.out / "market_prices.csv", index=False)
    news.to_csv(args.out / "news.csv", index=False)
    book_clients.to_csv(args.out / "book_clients.csv", index=False)
    ledger.to_csv(args.out / "book_transactions.csv", index=False)
    positions.to_csv(args.out / "book_positions.csv", index=False)
    print(f"Wrote market data ({len(prices):,} prices, {len(news)} news items) and brokerage book "
          f"({len(book_clients):,} clients, {len(ledger):,} ledger rows, {len(positions):,} positions); validation passed")


if __name__ == "__main__":
    main()