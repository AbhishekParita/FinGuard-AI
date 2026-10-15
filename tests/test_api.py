from fastapi.testclient import TestClient
from src.api.main import app
client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200 and response.json()["status"] == "ok"


def test_local_login_protects_data_routes():
    unauthenticated = TestClient(app)
    assert unauthenticated.get("/clients").status_code == 401
    response = client.post("/auth/login", json={"username": "demo", "password": "FinGuardDemo!"})
    assert response.status_code == 200
    assert client.get("/clients/CL99999/summary").status_code in (404, 503)
