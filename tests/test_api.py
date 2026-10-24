from fastapi.testclient import TestClient

from koora.api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_demo_insight():
    response = client.get("/api/demo/insight")
    assert response.status_code == 200
    assert response.json()["verification"]["status"] == "verified"


def test_web_served():
    response = client.get("/")
    assert response.status_code == 200
    assert "KOORA" in response.text
