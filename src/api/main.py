"""Local FastAPI backend for FinGuard AI."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, TypeVar

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from src.agents.orchestrator import answer_question
from src.data.local_db import create_session, delete_session, get_chat, init_db, save_chat, session_username, verify_user
from src.ml.predict_model import client_risk_summary, explain_transaction, score_transactions
from src.tools.financial import (
    client_profile,
    client_summary,
    list_clients,
    list_transactions,
    market_snapshot,
    portfolio_summary,
    transaction_aggregate,
)

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env", override=False)
app = FastAPI(title="FinGuard AI", version="0.2.0", description="Local synthetic-data financial analytics demo")
T = TypeVar("T")


class AssistantQuestion(BaseModel):
    client_id: str = Field(min_length=1, max_length=20)
    question: str = Field(min_length=1, max_length=500)


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=1, max_length=200)


@app.on_event("startup")
def initialize_local_db() -> None:
    init_db()


@app.middleware("http")
async def require_local_login(request: Request, call_next):
    public = {"/", "/health", "/config", "/auth/me", "/auth/login", "/auth/logout", "/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}
    username = session_username(request.cookies.get("finguard_session"))
    request.state.username = username
    if request.url.path not in public and not username:
        return JSONResponse({"detail": "Please sign in to the local demo."}, status_code=401)
    return await call_next(request)


def _run_data(operation: Callable[..., T], *args, **kwargs) -> T:
    try:
        return operation(*args, **kwargs)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Client not found.") from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "finguard-ai-local"}


@app.get("/", response_class=HTMLResponse)
def api_home() -> str:
    return '<html><body><p>FinGuard API is running. Open the dashboard at <a href="http://127.0.0.1:5173">http://127.0.0.1:5173</a>.</p></body></html>'


@app.get("/auth/me")
def auth_me(request: Request) -> dict:
    return {"authenticated": bool(request.state.username), "username": request.state.username}


@app.post("/auth/login")
def auth_login(payload: LoginRequest):
    if not verify_user(payload.username, payload.password):
        raise HTTPException(status_code=401, detail="Username or password is incorrect.")
    token, max_age = create_session(payload.username)
    response = JSONResponse({"authenticated": True, "username": payload.username})
    response.set_cookie("finguard_session", token, httponly=True, samesite="strict", max_age=max_age, path="/")
    return response


@app.post("/auth/logout")
def auth_logout(request: Request):
    delete_session(request.cookies.get("finguard_session"))
    response = JSONResponse({"authenticated": False})
    response.delete_cookie("finguard_session", path="/")
    return response


@app.get("/config")
def config() -> dict[str, bool | str]:
    return {
        "llm_enabled": bool(os.getenv("HF_TOKEN")),
        "llm_model": os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        "data_mode": "synthetic local datasets",
    }


@app.get("/clients")
def clients() -> dict[str, list[dict[str, str]]]:
    return {"clients": _run_data(list_clients)}


@app.get("/clients/{client_id}/summary")
def summary(client_id: str) -> dict:
    return _run_data(client_summary, client_id)


@app.get("/clients/{client_id}/profile")
def profile(client_id: str) -> dict:
    return _run_data(client_profile, client_id)


@app.get("/clients/{client_id}/portfolio")
def portfolio(client_id: str) -> dict:
    return _run_data(portfolio_summary, client_id)


@app.get("/clients/{client_id}/transactions")
def transactions(
    client_id: str,
    limit: int = Query(default=25, ge=1, le=100),
    category: str | None = None,
    predict: bool = False,
) -> dict:
    rows = _run_data(list_transactions, client_id, limit, category)
    response = {"transactions": rows}
    if predict:
        try:
            response["transactions"], response["model"] = score_transactions(rows)
        except FileNotFoundError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
    return response


@app.get("/clients/{client_id}/transactions/aggregate")
def aggregate(client_id: str, start_date: str | None = None, end_date: str | None = None) -> dict:
    return _run_data(transaction_aggregate, client_id, start_date, end_date)


@app.get("/clients/{client_id}/risk")
def risk(client_id: str) -> dict:
    try:
        return client_risk_summary(client_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Client not found.") from error


@app.get("/clients/{client_id}/risk/explanation")
def risk_explanation(client_id: str) -> dict:
    try:
        risk_data = client_risk_summary(client_id)
        if not risk_data["highest_risk_transactions"]:
            return {"contributions": [], "note": "This client has no transactions to explain."}
        return explain_transaction(risk_data["highest_risk_transactions"][0])
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Client not found.") from error
    except ImportError as error:
        raise HTTPException(status_code=503, detail="SHAP is not installed. Reinstall requirements.txt.") from error


@app.get("/market")
def market(symbol: str | None = None) -> dict:
    return _run_data(market_snapshot, symbol)


@app.post("/assistant")
def assistant(payload: AssistantQuestion, request: Request) -> dict:
    result = _run_data(answer_question, payload.question, payload.client_id)
    save_chat(request.state.username, payload.client_id, payload.question, result["answer"])
    return result


@app.get("/assistant/history/{client_id}")
def assistant_history(client_id: str, request: Request) -> dict:
    return {"messages": get_chat(request.state.username, client_id)}


@app.get("/model")
def model_info() -> dict:
    try:
        from src.ml.predict_model import load_model

        _, metadata = load_model()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {"model": "RandomForestClassifier", **metadata}
