"""Track E — Emotion Microscope API smoke tests."""

from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.serve import microscope_api


class _FakeSession:
    affect_vector = np.zeros(32, dtype=np.float32)
    traits = {"valence": 0.0}
    dominant_tone = "neutral"
    turn_index = 1
    affect_trajectory: list = []


class _FakeEngine:
    def __init__(self):
        self.session = _FakeSession()
        self._last_affect_source = "encoder:test"

    def generate_reply(self, user_text, *, return_introspection=False, **kwargs):
        return {
            "reply": "I hear you.",
            "traits": dict(self.session.traits),
            "dominant_tone": self.session.dominant_tone,
            "affect_source": self._last_affect_source,
            "encoder_source": "encoder:test",
            "gate_source": "gate:test",
            "gate_version": "v3.1_listener_ce_hardened",
            "gate_healthy": True,
            "amygdala_source": "snn:test",
            "collapse_detected": False,
            "collapse_score": 0.0,
            "recovered": False,
            "introspection": {
                "hooks_on": False,
                "rolling_kl_vs_hooks_off": 0.0,
                "turn_index": 1,
            },
        }

    def gate_health(self):
        return {
            "source": "trained",
            "version": "v3.1_listener_ce_hardened",
            "expected_version": "v3.1_listener_ce_hardened",
            "healthy": True,
            "warning": None,
        }

    def cleanup(self):
        pass


@pytest.fixture
def client():
    microscope_api._sessions.clear()
    microscope_api.set_engine_factory(_FakeEngine)
    yield TestClient(microscope_api.app)
    microscope_api._sessions.clear()
    microscope_api.set_engine_factory(None)


def test_state_endpoint(client):
    client.post("/chat", json={"message": "hello", "session_id": "s1"})
    r = client.get("/state/s1")
    assert r.status_code == 200
    body = r.json()
    assert "source" in body
    assert body["ok"] is True


def test_chat_returns_introspection(client):
    r = client.post("/chat", json={"message": "I feel sad today.", "session_id": "s2"})
    assert r.status_code == 200
    body = r.json()
    for key in ("reply", "source", "affect_vector", "introspection"):
        assert key in body


def test_chat_returns_collapse_and_gate_fields(client):
    """Phase 5 (docs/chat_hardening_plan.md): collapse guard + gate
    provenance surfaced in the /chat response."""
    r = client.post("/chat", json={"message": "I feel sad today.", "session_id": "s2b"})
    assert r.status_code == 200
    body = r.json()
    for key in (
        "collapse_detected",
        "collapse_score",
        "recovered",
        "gate_version",
        "gate_healthy",
    ):
        assert key in body
    assert body["collapse_detected"] is False
    assert body["gate_healthy"] is True


def test_health_endpoint_reports_gate_provenance(client):
    client.post("/chat", json={"message": "hello", "session_id": "s4"})
    r = client.get("/health/s4")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["gate"]["healthy"] is True
    assert body["gate"]["source"] == "trained"


def test_health_endpoint_unknown_session():
    from src.serve import microscope_api

    microscope_api._sessions.clear()
    from fastapi.testclient import TestClient

    client = TestClient(microscope_api.app)
    r = client.get("/health/no-such-session")
    assert r.status_code == 200
    assert r.json()["ok"] is False


def test_reset_clears_session(client):
    client.post("/chat", json={"message": "hi", "session_id": "s3"})
    r = client.post("/reset/s3")
    assert r.status_code == 200
    assert client.get("/state/s3").json()["ok"] is False
