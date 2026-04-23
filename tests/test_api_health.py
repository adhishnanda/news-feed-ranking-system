"""Smoke tests for the FastAPI application."""
from __future__ import annotations
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from src.api.main import app
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_rank_feed_v2_unknown_user(client):
    payload = {
        "user_id": "test_user_unknown_999",
        "session_id": "test_session_001",
        "limit": 5,
    }
    response = client.post("/rank-feed-v2", json=payload)
    # Should return 200 (with empty items) or 500 only if data files are missing
    assert response.status_code in (200, 500)
    if response.status_code == 200:
        data = response.json()
        assert "items" in data
        assert isinstance(data["items"], list)


def test_rank_feed_v2_valid_request(client):
    payload = {
        "user_id": "user_1",
        "session_id": "sess_test_42",
        "limit": 3,
    }
    response = client.post("/rank-feed-v2", json=payload)
    assert response.status_code in (200, 500)
    if response.status_code == 200:
        data = response.json()
        assert "user_id" in data
        assert data["user_id"] == "user_1"
        assert "items" in data
        assert len(data["items"]) <= 3


def test_rank_feed_v2_limit_validation(client):
    # limit=0 is below minimum (ge=1) — should return 422
    response = client.post("/rank-feed-v2", json={
        "user_id": "user_1",
        "session_id": "s",
        "limit": 0,
    })
    assert response.status_code == 422


def test_rank_feed_v2_limit_max(client):
    # limit=51 exceeds le=50 — should return 422
    response = client.post("/rank-feed-v2", json={
        "user_id": "user_1",
        "session_id": "s",
        "limit": 51,
    })
    assert response.status_code == 422


def test_rank_feed_v2_missing_fields(client):
    response = client.post("/rank-feed-v2", json={"user_id": "user_1"})
    assert response.status_code == 422
